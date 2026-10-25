"""Offline document parsers.

Two shapes of input:
  * documents (PDF, DOCX, PPTX, HTML, text, images, email, subtitles, code)
    -> parse_file() -> pages of text (+ structured tables);
  * record files (CSV / TSV / XLSX: alarm logs, work-order exports, tag lists)
    -> iter_records() -> streamed rows, never loaded fully into memory.

Default parsers are lightweight (pdfplumber, python-docx, Tesseract OCR for
scanned pages). IBM Docling (heavier, better layout/table structure) is used
only when RAG_USE_DOCLING=true and installed in the worker image.
Nothing here touches the network.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from email import message_from_binary_file, policy
from typing import Iterator, List, Optional, Tuple

from app.config import get_settings

log = logging.getLogger(__name__)

try:
    from app.rag.table_engine import markdown_to_rich_table, stitch_continuation_tables
    TABLE_ENGINE = True
except Exception:  # pragma: no cover
    TABLE_ENGINE = False


@dataclass
class PageContent:
    page_num: int
    text: str = ""
    rich_tables: List[object] = field(default_factory=list)  # table_engine.RichTable
    tables: List[str] = field(default_factory=list)           # markdown (only when no RichTable)


@dataclass
class ParseResult:
    pages: List[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
PPTX_EXT = {".pptx"}
HTML_EXT = {".html", ".htm"}
TEXT_EXT = {".txt", ".text", ".md", ".log", ".json", ".xml", ".yaml", ".yml", ".ini", ".cfg"}
CODE_EXT = {".st", ".scl", ".awl", ".py", ".js", ".c", ".h", ".cpp", ".cs", ".sql", ".sh", ".ps1"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
SUBTITLE_EXT = {".srt", ".vtt", ".ass", ".ssa"}
EMAIL_EXT = {".eml"}
RECORD_EXT = {".csv", ".tsv", ".xlsx", ".xlsm"}
ARCHIVE_EXT = {".zip", ".tar", ".tgz", ".gz"}
DOCUMENT_EXT = PDF_EXT | DOCX_EXT | PPTX_EXT | HTML_EXT | TEXT_EXT | CODE_EXT | IMAGE_EXT | SUBTITLE_EXT | EMAIL_EXT
SUPPORTED_EXTENSIONS = DOCUMENT_EXT | RECORD_EXT | ARCHIVE_EXT

_SUPERSCRIPTS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "½": "1/2", "¼": "1/4", "¾": "3/4", "®": "(R)", "™": "(TM)", " ": " ",
})


def normalize_text(text: str) -> str:
    """Normalise unicode noise so exact codes/values match (EQL40200D³ -> EQL40200D3)."""
    if not text:
        return ""
    text = text.translate(_SUPERSCRIPTS).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extension(path: str) -> str:
    name = path.lower()
    if name.endswith(".tar.gz"):
        return ".tgz"
    return os.path.splitext(name)[1]


def is_supported_file(path: str) -> bool:
    return extension(path) in SUPPORTED_EXTENSIONS


def is_record_file(path: str) -> bool:
    return extension(path) in RECORD_EXT


def get_file_type(path: str) -> str:
    ext = extension(path)
    for name, group in (("pdf", PDF_EXT), ("docx", DOCX_EXT), ("pptx", PPTX_EXT), ("html", HTML_EXT),
                        ("text", TEXT_EXT), ("code", CODE_EXT), ("image", IMAGE_EXT), ("subtitle", SUBTITLE_EXT),
                        ("email", EMAIL_EXT), ("archive", ARCHIVE_EXT)):
        if ext in group:
            return name
    if ext in {".csv", ".tsv"}:
        return "csv"
    if ext in {".xlsx", ".xlsm"}:
        return "xlsx"
    return "unknown"


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _rows_to_markdown(rows: List[List[object]]) -> str:
    rows = [[("" if c is None else str(c)).replace("\n", " ").replace("|", "/").strip() for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _add_table(page: PageContent, md: str, table_id: str, section: str) -> None:
    if not md:
        return
    rich = markdown_to_rich_table(md, table_id=table_id, section_title=section, page_no=page.page_num) if TABLE_ENGINE else None
    if rich is not None and rich.data_rows:
        page.rich_tables.append(rich)
    else:
        page.tables.append(md)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def _ocr_available() -> bool:
    return get_settings().ocr_enabled and shutil.which("tesseract") is not None


def _ocr_image(img) -> str:
    import pytesseract
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    try:
        osd = pytesseract.image_to_osd(img)
        angle = int(re.search(r"(?<=Rotate: )\d+", osd).group(0))
        if angle:
            img = img.rotate(-angle, expand=True)
    except Exception:
        pass  # too little text for orientation detection
    return pytesseract.image_to_string(img)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _outside(bboxes):
    """pdfplumber object filter: keep characters that are not inside any table."""
    def keep(obj) -> bool:
        if obj.get("object_type") != "char":
            return True
        x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
        return not any(b[0] - 1 <= x0 and x1 <= b[2] + 1 and b[1] - 1 <= top and bottom <= b[3] + 1 for b in bboxes)
    return keep


def _strip_running_lines(pages: List[PageContent]) -> None:
    """Remove running headers/footers: lines (digits ignored) repeated at the top/bottom of most pages."""
    if len(pages) < 3:
        return
    from collections import Counter
    counts: Counter = Counter()
    for pg in pages:
        lines = [l.strip() for l in pg.text.splitlines() if l.strip()]
        edge = set(lines[:2] + lines[-2:])
        counts.update(re.sub(r"\d+", "#", l) for l in edge)
    running = {k for k, v in counts.items() if v >= max(3, 0.5 * len(pages))}
    if not running:
        return
    for pg in pages:
        pg.text = "\n".join(l for l in pg.text.splitlines() if re.sub(r"\d+", "#", l.strip()) not in running)


def _last_heading(text: str) -> str:
    from app.rag.chunking import _heading  # local import: chunking does not import parsers
    found = ""
    for line in text.splitlines():
        title = _heading(line)
        if title:
            found = title
    return found


def _parse_pdf(path: str) -> ParseResult:
    import pdfplumber
    pages: List[PageContent] = []
    ocr_pages = 0
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages, 1):
            page = PageContent(page_num=i)
            try:
                tables = p.find_tables()
            except Exception as exc:
                log.debug("table detection failed on page %d: %s", i, exc)
                tables = []
            # Read body text outside table areas (tables are read cell by cell below),
            # otherwise multi-column table text comes out interleaved and garbled.
            body = p.filter(_outside([t.bbox for t in tables])) if tables else p
            text = body.extract_text() or ""
            page_area = float(p.width * p.height) or 1.0
            # A sizeable raster image (scanned insert, photographed nameplate, ...) is OCR'd too.
            big_image = any((im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) > 0.08 * page_area for im in p.images)
            if (len(text.strip()) < 40 or (big_image and len(text.strip()) < 600)) and _ocr_available():
                try:  # scanned page / scanned insert -> OCR at 200 dpi
                    ocr = _ocr_image(p.to_image(resolution=200).original)
                    text = ocr if len(text.strip()) < 40 else f"{text}\n{ocr}"
                    ocr_pages += 1
                except Exception as exc:
                    log.warning("OCR failed on %s page %d: %s", path, i, exc)
            page.text = normalize_text(text)
            for t_idx, table in enumerate(tables):
                try:
                    # The table belongs to the last heading printed above it on this page.
                    above = p.crop((0, 0, p.width, max(1, table.bbox[1]))).filter(_outside([t.bbox for t in tables]))
                    section = _last_heading(above.extract_text() or "")
                    _add_table(page, _rows_to_markdown(table.extract()), f"p{i}_t{t_idx}", section)
                except Exception as exc:
                    log.debug("table extraction failed on page %d: %s", i, exc)
            pages.append(page)
            p.flush_cache()  # keep memory flat on large manuals
    _strip_running_lines(pages)
    if TABLE_ENGINE:
        try:
            stitched = stitch_continuation_tables({pg.page_num: pg.rich_tables for pg in pages if pg.rich_tables})
            for pg in pages:
                if pg.page_num in stitched:
                    pg.rich_tables = stitched[pg.page_num]
        except Exception as exc:
            log.debug("table stitching failed: %s", exc)
    return ParseResult(pages=pages, metadata={"parser": "pdfplumber", "ocr_pages": ocr_pages})


def _parse_with_docling(path: str) -> ParseResult:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    opts.do_ocr = _ocr_available()
    if opts.do_ocr:
        try:
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions
            opts.ocr_options = TesseractCliOcrOptions()
        except ImportError:
            pass
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = converter.convert(path).document
    pages = {}
    section = ""
    for t_idx, (item, _level) in enumerate(doc.iterate_items()):
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else 1
        page = pages.setdefault(page_no, PageContent(page_num=page_no))
        kind = type(item).__name__
        if kind == "SectionHeaderItem":
            section = normalize_text(item.text)
            page.text += f"\n## {section}\n"
        elif kind == "TableItem":
            _add_table(page, normalize_text(item.export_to_markdown(doc)), f"p{page_no}_t{t_idx}", section)
        elif hasattr(item, "text") and item.text:
            page.text += normalize_text(item.text) + "\n\n"
    return ParseResult(pages=[pages[k] for k in sorted(pages)], metadata={"parser": "docling"})


def _docling_enabled() -> bool:
    if not get_settings().use_docling:
        return False
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        log.warning("RAG_USE_DOCLING=true but docling is not installed; using lightweight parsers")
        return False


# ---------------------------------------------------------------------------
# Office / HTML / text
# ---------------------------------------------------------------------------
def _parse_docx(path: str) -> ParseResult:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(path)
    page = PageContent(page_num=1)
    lines: List[str] = []
    section = ""
    t_idx = 0
    for block in doc.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(block, doc)
            text = normalize_text(para.text)
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            if m or style == "Title":
                level = int(m.group(1)) if m else 1
                section = text
                lines.append("\n" + "#" * min(level + 1, 6) + " " + text + "\n")
            else:
                lines.append(text)
        elif tag == "tbl":
            table = Table(block, doc)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            _add_table(page, _rows_to_markdown(rows), f"docx_t{t_idx}", section)
            t_idx += 1
    page.text = "\n".join(lines)
    return ParseResult(pages=[page], metadata={"parser": "python-docx"})


def _parse_pptx(path: str) -> ParseResult:
    from pptx import Presentation
    pages = []
    for i, slide in enumerate(Presentation(path).slides, 1):
        page = PageContent(page_num=i)
        texts = []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[c.text for c in r.cells] for r in shape.table.rows]
                _add_table(page, _rows_to_markdown(rows), f"s{i}_t{len(page.rich_tables)}", "")
            elif getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text)
        if texts:
            page.text = f"## Slide {i}: {normalize_text(texts[0])[:120]}\n" + normalize_text("\n".join(texts[1:]))
        pages.append(page)
    return ParseResult(pages=pages, metadata={"parser": "python-pptx"})


def _parse_html(path: str) -> ParseResult:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    page = PageContent(page_num=1)
    for i, table in enumerate(soup.find_all("table")):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        _add_table(page, _rows_to_markdown(rows), f"html_t{i}", "")
        table.decompose()
    for level in range(1, 5):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n{'#' * (level + 1)} {h.get_text(' ', strip=True)}\n")
    page.text = normalize_text(soup.get_text("\n"))
    return ParseResult(pages=[page], metadata={"parser": "bs4"})


def _parse_text(path: str) -> ParseResult:
    text = normalize_text(_read_text(path))
    if extension(path) in CODE_EXT:
        text = f"## {os.path.basename(path)}\n{text}"
    return ParseResult(pages=[PageContent(page_num=1, text=text)], metadata={"parser": "text"})


def _parse_image(path: str) -> ParseResult:
    if not _ocr_available():
        return ParseResult(success=False, error="OCR (tesseract) not available for images")
    from PIL import Image, ImageOps
    with Image.open(path) as img:
        text = _ocr_image(ImageOps.exif_transpose(img))
    return ParseResult(pages=[PageContent(page_num=1, text=normalize_text(text))], metadata={"parser": "tesseract"})


def _parse_subtitles(path: str) -> ParseResult:
    lines = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line or line.upper().startswith(("WEBVTT", "[SCRIPT", "DIALOGUE:")):
            continue
        lines.append(re.sub(r"<[^>]+>|\{[^}]*\}", "", line))
    return ParseResult(pages=[PageContent(page_num=1, text=normalize_text("\n".join(lines)))], metadata={"parser": "subtitles"})


def _parse_email(path: str) -> ParseResult:
    with open(path, "rb") as f:
        msg = message_from_binary_file(f, policy=policy.default)
    body = msg.get_body(preferencelist=("plain", "html"))
    content = body.get_content() if body else ""
    if body is not None and body.get_content_type() == "text/html":
        from bs4 import BeautifulSoup
        content = BeautifulSoup(content, "html.parser").get_text("\n")
    header = f"## Email: {msg.get('subject', '(no subject)')}\nFrom: {msg.get('from', '')}\nDate: {msg.get('date', '')}\n"
    return ParseResult(pages=[PageContent(page_num=1, text=header + normalize_text(content))], metadata={"parser": "email"})


def parse_file(path: str) -> ParseResult:
    """Parse a non-record document into pages. Never raises; check `.success`."""
    if not os.path.isfile(path):
        return ParseResult(success=False, error=f"file not found: {path}")
    ext = extension(path)
    try:
        if ext in PDF_EXT:
            if _docling_enabled():
                try:
                    return _parse_with_docling(path)
                except Exception as exc:
                    log.warning("docling failed on %s (%s); falling back to pdfplumber", path, exc)
            return _parse_pdf(path)
        if ext in DOCX_EXT:
            return _parse_docx(path)
        if ext in PPTX_EXT:
            return _parse_pptx(path)
        if ext in HTML_EXT:
            return _parse_html(path)
        if ext in TEXT_EXT | CODE_EXT:
            return _parse_text(path)
        if ext in IMAGE_EXT:
            return _parse_image(path)
        if ext in SUBTITLE_EXT:
            return _parse_subtitles(path)
        if ext in EMAIL_EXT:
            return _parse_email(path)
        return ParseResult(success=False, error=f"unsupported document type: {ext}")
    except Exception as exc:
        log.exception("parse failed: %s", path)
        return ParseResult(success=False, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Record files (streamed)
# ---------------------------------------------------------------------------
def iter_records(path: str) -> Iterator[Tuple[str, List[str], List[str], int]]:
    """Yield (sheet, header, row, row_number) for CSV/TSV/XLSX without loading the file."""
    ext = extension(path)
    if ext in {".csv", ".tsv"}:
        text_stream = io.TextIOWrapper(open(path, "rb"), encoding="utf-8-sig", errors="replace", newline="")
        with text_stream as f:
            sample = f.read(65536)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|") if ext == ".csv" else csv.excel_tab
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            header = None
            for n, row in enumerate(reader, 1):
                if not any(c.strip() for c in row):
                    continue
                if header is None:
                    header = [normalize_text(c) or f"Column {i + 1}" for i, c in enumerate(row)]
                    continue
                yield "", header, [normalize_text(c) for c in row], n
    elif ext in {".xlsx", ".xlsm"}:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for ws in wb.worksheets:
                header = None
                for n, row in enumerate(ws.iter_rows(values_only=True), 1):
                    cells = ["" if c is None else normalize_text(str(c)) for c in row]
                    if not any(cells):
                        continue
                    if header is None:
                        header = [c or f"Column {i + 1}" for i, c in enumerate(cells)]
                        continue
                    yield ws.title, header, cells, n
        finally:
            wb.close()
    else:
        raise ValueError(f"not a record file: {path}")


def extract_archive(path: str, dest_dir: str, max_bytes: int = 1 << 30) -> List[str]:
    """Safely extract supported files from an archive (zip-slip and zip-bomb guarded)."""
    import tarfile
    import zipfile

    dest_dir = os.path.realpath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    out, total = [], 0

    def safe(name: str) -> Optional[str]:
        target = os.path.realpath(os.path.join(dest_dir, name))
        if not target.startswith(dest_dir + os.sep) or not is_supported_file(name) or extension(name) in ARCHIVE_EXT:
            return None
        return target

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                target = safe(info.filename)
                if info.is_dir() or target is None or os.path.basename(info.filename).startswith("."):
                    continue
                total += info.file_size
                if total > max_bytes:
                    raise ValueError("archive exceeds extraction size limit")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(target)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                target = safe(member.name)
                if not member.isfile() or target is None:
                    continue
                total += member.size
                if total > max_bytes:
                    raise ValueError("archive exceeds extraction size limit")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(target)
    else:
        raise ValueError("unsupported or corrupt archive")
    return out

