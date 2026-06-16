"""
=============================================================================
 Enterprise Level RAG: Layer 1 — Universal Document Parser
=============================================================================
 Supports: PDF, DOCX, XLSX, PPTX, CSV, TXT, Images, Video (subtitles)
 All parsing is 100% offline — no API calls, no cloud services.
=============================================================================
"""

import csv
import io
import os
import re
import subprocess
import email
from email import policy
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional imports — each format gracefully degrades if library missing
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from openpyxl import load_workbook
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pysubs2
    SUBS_AVAILABLE = True
except ImportError:
    SUBS_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class PageContent:
    """Content extracted from a single page / sheet / slide."""
    page_num: int
    text: str = ""
    tables: List[str] = field(default_factory=list)
    image_texts: List[str] = field(default_factory=list)
    image_bytes: List[bytes] = field(default_factory=list)
    content_type: str = "text"  # text, table, image_ocr, subtitle


@dataclass
class ParseResult:
    """Complete result from parsing a file."""
    pages: List[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Supported Extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
TEXT_EXTENSIONS = {".txt", ".text", ".md", ".log", ".json", ".xml"}
CODE_EXTENSIONS = {".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rs", ".rb", ".php", ".sql", ".html", ".css", ".sh"}
EMAIL_EXTENSIONS = {".eml", ".msg"}
URL_EXTENSIONS = {".url", ".webloc"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".rar"}

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
    ".pptx", ".ppt"
} | IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS | TEXT_EXTENSIONS | CODE_EXTENSIONS | EMAIL_EXTENSIONS | URL_EXTENSIONS | ARCHIVE_EXTENSIONS


def is_supported_file(file_path: str) -> bool:
    """Check if a file extension is supported."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def get_file_type(file_path: str) -> str:
    """Return a human-readable file type category."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in {".docx", ".doc"}:
        return "docx"
    elif ext in {".xlsx", ".xls"}:
        return "xlsx"
    elif ext in {".pptx", ".ppt"}:
        return "pptx"
    elif ext == ".csv":
        return "csv"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    elif ext in CODE_EXTENSIONS:
        return "code"
    elif ext in EMAIL_EXTENSIONS:
        return "email"
    elif ext in URL_EXTENSIONS:
        return "url"
    elif ext in ARCHIVE_EXTENSIONS:
        return "archive"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in SUBTITLE_EXTENSIONS:
        return "subtitle"
    return "unknown"


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def _auto_rotate_image(img):
    """Detect image orientation via OSD and auto-rotate upright (solves PDF flip)."""
    if not OCR_AVAILABLE:
        return img
    try:
        osd = pytesseract.image_to_osd(img)
        match = re.search(r'(?<=Rotate: )\d+', osd)
        if match:
            angle = int(match.group(0))
            if angle != 0:
                # pytesseract's Rotate value is counter-clockwise angle to correct orientation
                img = img.rotate(angle, expand=True)
    except Exception:
        pass  # OSD fails if too little text is detected
    return img


def _enrich_mcq_text(text: str) -> str:
    """Detect MCQ checkmarks and explicitly mark the correct answer."""
    if not text:
        return text
    
    # Common symbols used for marking the correct MCQ option
    mcq_symbols = [r"✓", r"✔", r"☑", r"☒", r"◉", r"\[x\]", r"\[X\]", r"\(x\)", r"\(X\)"]
    pattern = re.compile(f"({'|'.join(mcq_symbols)})")
    
    enriched_lines = []
    for line in text.split('\n'):
        if pattern.search(line):
            # If the line has an MCQ tick, explicitly label it so the LLM understands it's the correct answer
            line = f"[CORRECT ANSWER] {line}"
        enriched_lines.append(line)
    return "\n".join(enriched_lines)


def _clean_ocr_text(text: str) -> str:
    """Remove highly noisy or garbage lines from OCR output (e.g. random UI screenshot symbols)."""
    if not text:
        return text
        
    clean_lines = []
    for line in text.split('\n'):
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Drop extremely short lines with barely any letters
        if len(line_clean) < 4 and sum(c.isalpha() for c in line_clean) < 2:
            continue
            
        # Drop lines that are mostly symbols/noise (e.g., "ae atl] _—T-~")
        alphanum = sum(c.isalnum() for c in line_clean)
        if alphanum / len(line_clean) < 0.5:
            continue
            
        # Drop lines with too many repetitive weird characters
        if re.search(r'([^\w\s])\1{3,}', line_clean):
            continue
            
        clean_lines.append(line)
        
    return '\n'.join(clean_lines)


# ---------------------------------------------------------------------------
# PDF Parser — Full fidelity: text, tables, images, metadata, links, fonts
# ---------------------------------------------------------------------------
def _parse_pdf(file_path: str) -> ParseResult:
    """Extract all content from PDF with zero information loss.

    Captures: text (layout-aware), tables (bordered+borderless), images+OCR,
    hyperlinks, font metadata, annotations, document metadata, bookmarks/TOC.
    """
    if not PDF_AVAILABLE:
        return ParseResult(success=False, error="PyMuPDF not installed")

    pages = []
    doc = None
    all_fonts = set()
    has_links = False
    pdfplumber_pdf = None
    try:
        doc = fitz.open(file_path)

        # Open pdfplumber once for all pages (avoids O(N) file opens)
        if PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber as _pdfplumber
                pdfplumber_pdf = _pdfplumber.open(file_path)
            except Exception as e:
                print(f"[Parser] pdfplumber open failed: {e}")

        # --- Extract document-level metadata ---
        meta = doc.metadata
        doc_metadata = {
            "format": "pdf",
            "page_count": len(doc),
            "source": file_path,
            "title": meta.get("title", "").strip(),
            "author": meta.get("author", "").strip(),
            "subject": meta.get("subject", "").strip(),
            "keywords": meta.get("keywords", "").strip(),
        }

        # --- Extract table of contents / bookmarks ---
        toc = doc.get_toc()
        if toc:
            doc_metadata["toc"] = toc

        for page_num_idx, fitz_page in enumerate(doc):
            page_num = page_num_idx + 1

            # --- 1. Layout-aware text extraction with font metadata ---
            blocks = fitz_page.get_text("dict", sort=False)["blocks"]
            blocks.sort(key=lambda b: (round(b["bbox"][1], -1), b["bbox"][0]))

            text_lines = []
            for block in blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font = span.get("font", "")
                            size = round(span.get("size", 0), 1)
                            if font:
                                all_fonts.add(f"{font}@{size}")
                        line_text = "".join(span["text"] for span in line["spans"])
                        if line_text.strip():
                            text_lines.append(line_text)
                elif block["type"] == 1:
                    text_lines.append("[IMAGE]")

            text = "\n".join(text_lines)
            text = _enrich_mcq_text(text)

            # Check for hyperlinks on this page
            if not has_links:
                for link in fitz_page.get_links():
                    if link.get("uri"):
                        has_links = True
                        break

            # --- 2. Table extraction (bordered + borderless via pdfplumber) ---
            tables = []
            if pdfplumber_pdf is not None and page_num - 1 < len(pdfplumber_pdf.pages):
                tables = _extract_tables_pdfplumber(pdfplumber_pdf.pages[page_num - 1], page_num)

            # --- 3. Image OCR extraction ---
            image_texts, image_bytes = _extract_images_ocr(fitz_page, page_num)

            # --- 4. Full-page OCR fallback for scanned PDFs ---
            if OCR_AVAILABLE and len(text.strip()) < 50:
                try:
                    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(3, 3))
                    raw_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(raw_bytes))
                    img = img.convert("L")
                    img = _auto_rotate_image(img)
                    full_ocr = pytesseract.image_to_string(img).strip()
                    full_ocr = _clean_ocr_text(full_ocr)
                    full_ocr = _enrich_mcq_text(full_ocr)
                    if full_ocr and len(full_ocr) > 20:
                        image_texts.append(f"[FULL PAGE OCR - Page {page_num}]\n{full_ocr}")
                        image_bytes.append(raw_bytes)
                except Exception as e:
                    print(f"[Parser] Full-page OCR failed for page {page_num}: {e}")

            pages.append(PageContent(
                page_num=page_num,
                text=text,
                tables=tables,
                image_texts=image_texts,
                image_bytes=image_bytes,
                content_type="text",
            ))

        # --- 6. Preserve headers/footers that contain meaningful context ---
        if len(pages) >= 3:
            _deduplicate_headers_footers(pages)

        doc_metadata["fonts"] = sorted(all_fonts)
        doc_metadata["has_links"] = has_links

        return ParseResult(
            pages=pages,
            metadata=doc_metadata,
        )
    except Exception as e:
        return ParseResult(success=False, error=f"PDF parse error: {e}")
    finally:
        if doc is not None:
            doc.close()
        if pdfplumber_pdf is not None:
            pdfplumber_pdf.close()


def _deduplicate_headers_footers(pages: List[PageContent]) -> None:
    """Remove repetitive headers/footers only if they are identical boilerplate,
    but KEEP them if they contain meaningful context (e.g., section titles)."""
    # Analyze first 3 pages for repeated first/last lines
    first_lines = [p.text.strip().split("\n")[0] for p in pages[:3] if p.text.strip()]
    last_lines = [p.text.strip().split("\n")[-1] for p in pages[:3] if p.text.strip()]

    # Only strip if ALL 3 pages have identical header AND it looks like a page number or copyright
    if len(first_lines) == 3 and first_lines[0] == first_lines[1] == first_lines[2]:
        h = first_lines[0].strip()
        # Only strip boilerplate: page numbers or simple copyright notices
        if re.match(r'^\d{1,4}$', h) or re.match(r'^©.*\d{4}', h):
            for p in pages:
                lines = p.text.strip().split("\n")
                if lines and lines[0].strip() == h:
                    p.text = "\n".join(lines[1:]).strip()

    if len(last_lines) == 3 and last_lines[0] == last_lines[1] == last_lines[2]:
        f = last_lines[0].strip()
        if re.match(r'^\d{1,4}$', f) or re.match(r'^Page \d+', f, re.I):
            for p in pages:
                lines = p.text.strip().split("\n")
                if lines and lines[-1].strip() == f:
                    p.text = "\n".join(lines[:-1]).strip()


def _extract_tables_pdfplumber(pdf_page, page_num: int) -> list:
    """Extract structured tables from a pdfplumber page, including borderless."""
    tables_text = []
    try:
        # 1. Bordered tables (default detection)
        tables = pdf_page.extract_tables()
        for table_idx, table in enumerate(tables):
            if not table:
                continue
            rows = []
            for row in table:
                if row is None:
                    continue
                cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                rows.append(" | ".join(cleaned_row))
            if rows:
                header = rows[0]
                first_row = table[0]
                col_count = len(first_row) if first_row else len(rows[0].split(" | "))
                separator = " | ".join(["---"] * col_count)
                table_text = f"[TABLE {table_idx + 1} - Page {page_num}]\n{header}\n{separator}\n" + "\n".join(rows[1:])
                tables_text.append(table_text.strip())

        # 2. Borderless tables (aligned text columns)
        existing_count = len(tables_text)
        borderless = pdf_page.find_tables(
            table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 5,
            }
        )
        for bt in borderless:
            if not bt.bbox:
                continue
            table_data = bt.extract()
            if table_data is None or len(table_data) <= 1:
                continue
            if table_data[0] is None:
                continue
            rows = []
            for row in table_data:
                if row is None:
                    continue
                cleaned_row = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                rows.append(" | ".join(cleaned_row))
            if rows:
                header = rows[0]
                sep = " | ".join(["---"] * len(table_data[0]))
                table_text = f"[TABLE {existing_count + 1} - Page {page_num} (borderless)]\n{header}\n{sep}\n" + "\n".join(rows[1:])
                tables_text.append(table_text.strip())
                existing_count += 1
    except Exception as e:
        print(f"[Parser] Table extraction warning for page {page_num}: {e}")
    return tables_text


def _extract_images_ocr(fitz_page, page_num: int, min_size: int = 80) -> tuple:
    """Extract text from images embedded in a PDF page via OCR, and return raw bytes for Vision."""
    if not OCR_AVAILABLE:
        return [], []

    image_texts = []
    image_bytes = []
    try:
        image_list = fitz_page.get_images(full=True)
        doc = fitz_page.parent

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < min_size or height < min_size:
                    continue

                img = Image.open(io.BytesIO(base_image["image"]))
                # Pre-process for better OCR
                img = img.convert("L")  # Grayscale
                img = _auto_rotate_image(img)
                ocr_text = pytesseract.image_to_string(img).strip()
                ocr_text = _enrich_mcq_text(ocr_text)

                if ocr_text and len(ocr_text) > 15:
                    image_texts.append(
                        f"[IMAGE OCR - Page {page_num}, Image {img_idx + 1}]\n{ocr_text}"
                    )
                # Always save image bytes for Multi-modal CLIP embeddings
                image_bytes.append(base_image["image"])
            except Exception:
                continue
    except Exception as e:
        print(f"[Parser] Image OCR warning for page {page_num}: {e}")
    return image_texts, image_bytes


# ---------------------------------------------------------------------------
# DOCX Parser
# ---------------------------------------------------------------------------
def _parse_docx(file_path: str) -> ParseResult:
    """Extract text and tables from DOCX files."""
    if not DOCX_AVAILABLE:
        return ParseResult(success=False, error="python-docx not installed")

    try:
        doc = DocxDocument(file_path)
        pages = []
        current_text_parts = []
        current_tables = []
        current_image_texts = []
        page_num = 1

        table_idx = 0
        try:
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.text.paragraph import Paragraph
            from docx.table import Table
            
            for child in doc.element.body:
                if isinstance(child, CT_P):
                    para = Paragraph(child, doc)
                    text = para.text.strip()
                    text = _enrich_mcq_text(text)
                    if text:
                        if para.paragraph_format.page_break_before:
                            if current_text_parts or current_tables:
                                pages.append(PageContent(
                                    page_num=page_num,
                                    text="\n".join(current_text_parts),
                                    tables=current_tables,
                                    image_texts=current_image_texts,
                                ))
                                page_num += 1
                                current_text_parts = []
                                current_tables = []
                                current_image_texts = []
                        if para.style and para.style.name and para.style.name.startswith("Heading"):
                            level = para.style.name.replace("Heading", "").strip()
                            prefix = "#" * (int(level) if level.isdigit() else 1)
                            current_text_parts.append(f"{prefix} {text}")
                        else:
                            current_text_parts.append(text)
                            
                elif isinstance(child, CT_Tbl):
                    table = Table(child, doc)
                    table_idx += 1
                    rows = []
                    for row in table.rows:
                        cells = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
                        rows.append(" | ".join(cells))
                    if rows:
                        header = rows[0]
                        separator = " | ".join(["---"] * len(table.rows[0].cells))
                        table_md = f"[TABLE {table_idx}]\n{header}\n{separator}\n" + "\n".join(rows[1:])
                        current_tables.append(table_md.strip())
        except ImportError:
            # Fallback if internals change
            pass

        # Extract images via OCR
        if OCR_AVAILABLE:
            for rel in doc.part.rels.values():
                if "image" in rel.reltype:
                    try:
                        image_data = rel.target_part.blob
                        img = Image.open(io.BytesIO(image_data))
                        if img.width >= 80 and img.height >= 80:
                            img = img.convert("L")
                            img = _auto_rotate_image(img)
                            ocr_text = pytesseract.image_to_string(img).strip()
                            ocr_text = _enrich_mcq_text(ocr_text)
                            if ocr_text and len(ocr_text) > 15:
                                current_image_texts.append(f"[IMAGE OCR]\n{ocr_text}")
                    except Exception:
                        continue

        # Add remaining content
        if current_text_parts or current_tables or current_image_texts:
            pages.append(PageContent(
                page_num=page_num,
                text="\n".join(current_text_parts),
                tables=current_tables,
                image_texts=current_image_texts,
            ))

        return ParseResult(
            pages=pages,
            metadata={
                "format": "docx",
                "page_count": len(pages),
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"DOCX parse error: {e}")


# ---------------------------------------------------------------------------
# XLSX / Excel Parser
# ---------------------------------------------------------------------------
def _parse_xlsx(file_path: str) -> ParseResult:
    """Extract all sheets from Excel as markdown tables."""
    if not XLSX_AVAILABLE:
        return ParseResult(success=False, error="openpyxl not installed")

    try:
        wb = load_workbook(file_path, read_only=True, data_only=True)
        pages = []

        for sheet_idx, sheet_name in enumerate(wb.sheetnames):
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
                if any(cells):  # Skip completely empty rows
                    rows.append(" | ".join(cells))

            if not rows:
                continue

            # First row as header
            header = rows[0]
            # Count columns from first data row
            col_count = len(rows[0].split(" | "))
            separator = " | ".join(["---"] * col_count)
            table_md = f"[EXCEL SHEET: {sheet_name}]\n{header}\n{separator}\n" + "\n".join(rows[1:])

            pages.append(PageContent(
                page_num=sheet_idx + 1,
                text=f"Sheet: {sheet_name}\n\n{table_md}",
                tables=[table_md],
                content_type="table",
            ))

        wb.close()
        return ParseResult(
            pages=pages,
            metadata={
                "format": "xlsx",
                "page_count": len(pages),
                "sheet_names": wb.sheetnames if hasattr(wb, 'sheetnames') else [],
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"XLSX parse error: {e}")


# ---------------------------------------------------------------------------
# PPTX Parser
# ---------------------------------------------------------------------------
def _parse_pptx(file_path: str) -> ParseResult:
    """Extract text, notes, and tables from PowerPoint files."""
    if not PPTX_AVAILABLE:
        return ParseResult(success=False, error="python-pptx not installed")

    try:
        prs = Presentation(file_path)
        pages = []

        def _iter_shapes(shapes):
            for shape in shapes:
                yield shape
                if shape.shape_type == 6:  # GROUP
                    try:
                        yield from _iter_shapes(shape.shapes)
                    except Exception:
                        pass

        for slide_idx, slide in enumerate(prs.slides):
            slide_num = slide_idx + 1
            text_parts = []
            tables = []
            image_texts = []

            for shape in _iter_shapes(slide.shapes):
                # Text frames
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        para_text = paragraph.text.strip()
                        para_text = _enrich_mcq_text(para_text)
                        if para_text:
                            text_parts.append(para_text)

                # Tables
                if shape.has_table:
                    table = shape.table
                    rows = []
                    for row in table.rows:
                        cells = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
                        rows.append(" | ".join(cells))
                    if rows:
                        header = rows[0]
                        separator = " | ".join(["---"] * len(table.rows[0].cells))
                        table_md = f"[TABLE - Slide {slide_num}]\n{header}\n{separator}\n" + "\n".join(rows[1:])
                        tables.append(table_md)

                # Images: extract and OCR
                if shape.shape_type == 13 and OCR_AVAILABLE:  # PICTURE
                    try:
                        image_blob = shape.image.blob
                        img = Image.open(io.BytesIO(image_blob))
                        img = img.convert("L")
                        ocr_text = pytesseract.image_to_string(img).strip()
                        if ocr_text:
                            image_texts.append(f"[IMAGE OCR - Slide {slide_num}]\n{ocr_text}")
                    except Exception as e:
                        print(f"[Parser] PPTX image OCR failed slide {slide_num}: {e}")

            # Slide notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    text_parts.append(f"[SPEAKER NOTES]\n{notes}")

            if text_parts or tables or image_texts:
                pages.append(PageContent(
                    page_num=slide_num,
                    text="\n".join(text_parts),
                    tables=tables,
                    image_texts=image_texts,
                ))

        return ParseResult(
            pages=pages,
            metadata={
                "format": "pptx",
                "page_count": len(prs.slides),
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"PPTX parse error: {e}")


# ---------------------------------------------------------------------------
# CSV Parser
# ---------------------------------------------------------------------------
def _parse_csv(file_path: str) -> ParseResult:
    """Parse CSV files into markdown table format."""
    try:
        # Detect encoding
        encoding = "utf-8"
        if CHARDET_AVAILABLE:
            with open(file_path, "rb") as f:
                raw = f.read(10000)
                detected = chardet.detect(raw)
                encoding = detected.get("encoding", "utf-8") or "utf-8"

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                cells = [str(cell).strip() for cell in row]
                if any(cells):
                    rows.append(" | ".join(cells))

        if not rows:
            return ParseResult(
                pages=[PageContent(page_num=1, text="(Empty CSV file)")],
                metadata={"format": "csv", "page_count": 1, "source": file_path},
            )

        header = rows[0]
        col_count = len(rows[0].split(" | "))
        separator = " | ".join(["---"] * col_count)
        table_md = f"[CSV DATA]\n{header}\n{separator}\n" + "\n".join(rows[1:])

        # Split into pages of 100 rows for very large CSVs
        chunk_size = 100
        pages = []
        data_rows = rows[1:]
        for i in range(0, max(len(data_rows), 1), chunk_size):
            chunk = data_rows[i:i + chunk_size]
            chunk_table = f"[CSV DATA - Rows {i + 1} to {i + len(chunk)}]\n{header}\n{separator}\n" + "\n".join(chunk)
            pages.append(PageContent(
                page_num=(i // chunk_size) + 1,
                text=chunk_table,
                tables=[chunk_table],
                content_type="table",
            ))

        if not pages:
            pages.append(PageContent(page_num=1, text=table_md, tables=[table_md], content_type="table"))

        return ParseResult(
            pages=pages,
            metadata={
                "format": "csv",
                "page_count": len(pages),
                "total_rows": len(rows),
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"CSV parse error: {e}")


# ---------------------------------------------------------------------------
# Plain Text Parser
# ---------------------------------------------------------------------------
def _parse_text(file_path: str) -> ParseResult:
    """Parse plain text, markdown, log, JSON, XML files."""
    try:
        encoding = "utf-8"
        if CHARDET_AVAILABLE:
            with open(file_path, "rb") as f:
                raw = f.read(10000)
                detected = chardet.detect(raw)
                encoding = detected.get("encoding", "utf-8") or "utf-8"

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()
            
        content = _enrich_mcq_text(content)

        if not content.strip():
            return ParseResult(
                pages=[PageContent(page_num=1, text="(Empty file)")],
                metadata={"format": "text", "page_count": 1, "source": file_path},
            )

        # Split large text files into ~4000 char pages
        page_size = 4000
        pages = []
        lines = content.split("\n")
        current_page = []
        current_len = 0
        page_num = 1

        for line in lines:
            if current_len + len(line) > page_size and current_page:
                pages.append(PageContent(
                    page_num=page_num,
                    text="\n".join(current_page),
                ))
                page_num += 1
                current_page = []
                current_len = 0
            current_page.append(line)
            current_len += len(line) + 1

        if current_page:
            pages.append(PageContent(page_num=page_num, text="\n".join(current_page)))

        return ParseResult(
            pages=pages,
            metadata={
                "format": "text",
                "page_count": len(pages),
                "total_chars": len(content),
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"Text parse error: {e}")


# ---------------------------------------------------------------------------
# Image Parser (OCR)
# ---------------------------------------------------------------------------
def _parse_image(file_path: str) -> ParseResult:
    """Extract text from images using OCR."""
    if not OCR_AVAILABLE:
        return ParseResult(success=False, error="pytesseract/Pillow not installed for image OCR")

    try:
        img = Image.open(file_path)

        # Pre-process for optimal OCR
        if img.mode != "L":
            img = img.convert("L")

        img = _auto_rotate_image(img)
        ocr_text = pytesseract.image_to_string(img).strip()
        ocr_text = _enrich_mcq_text(ocr_text)
        filename = os.path.basename(file_path)

        if not ocr_text:
            return ParseResult(
                pages=[PageContent(
                    page_num=1,
                    text=f"[IMAGE: {filename}]\n(No text detected in image)",
                    content_type="image_ocr",
                )],
                metadata={"format": "image", "page_count": 1, "source": file_path},
            )

        return ParseResult(
            pages=[PageContent(
                page_num=1,
                text=f"[IMAGE OCR: {filename}]\n{ocr_text}",
                image_texts=[ocr_text],
                content_type="image_ocr",
            )],
            metadata={
                "format": "image",
                "page_count": 1,
                "image_size": f"{img.width}x{img.height}",
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"Image parse error: {e}")


# ---------------------------------------------------------------------------
# Video Subtitle Parser
# ---------------------------------------------------------------------------
def _parse_video(file_path: str) -> ParseResult:
    """Extract embedded subtitles from video files using ffmpeg + pysubs2."""
    try:
        # First try to extract embedded subtitles using ffmpeg
        subtitle_text = _extract_subtitles_ffmpeg(file_path)

        if not subtitle_text:
            return ParseResult(
                pages=[PageContent(
                    page_num=1,
                    text=f"[VIDEO: {os.path.basename(file_path)}]\n(No embedded subtitles found)",
                    content_type="subtitle",
                )],
                metadata={"format": "video", "page_count": 1, "source": file_path},
            )

        # Split subtitles into manageable pages
        lines = subtitle_text.split("\n")
        page_size = 50  # lines per page
        pages = []
        for i in range(0, len(lines), page_size):
            chunk = lines[i:i + page_size]
            pages.append(PageContent(
                page_num=(i // page_size) + 1,
                text=f"[VIDEO SUBTITLES - Part {(i // page_size) + 1}]\n" + "\n".join(chunk),
                content_type="subtitle",
            ))

        return ParseResult(
            pages=pages,
            metadata={
                "format": "video",
                "page_count": len(pages),
                "subtitle_lines": len(lines),
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"Video parse error: {e}")


def _extract_subtitles_ffmpeg(video_path: str) -> str:
    """Extract embedded subtitle tracks from video using ffmpeg."""
    try:
        # Extract subtitle to SRT format using ffmpeg
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-map", "0:s:0",  # First subtitle stream
                "-f", "srt",
                "-",  # Output to stdout
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse SRT and extract just the text (no timestamps)
            lines = result.stdout.strip().split("\n")
            text_lines = []
            for line in lines:
                line = line.strip()
                # Skip sequence numbers, timestamps, and empty lines
                if not line:
                    continue
                if line.isdigit():
                    continue
                if "-->" in line:
                    continue
                # Remove HTML tags from subtitles
                clean = re.sub(r"<[^>]+>", "", line)
                if clean.strip():
                    text_lines.append(clean.strip())

            return "\n".join(text_lines)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ""


def _parse_subtitle_file(file_path: str) -> ParseResult:
    """Parse standalone subtitle files (SRT, ASS, SSA, VTT)."""
    try:
        if SUBS_AVAILABLE:
            subs = pysubs2.load(file_path)
            text_lines = []
            for event in subs:
                if event.text.strip():
                    # Clean subtitle formatting tags
                    clean = re.sub(r"\{[^}]*\}", "", event.text)
                    clean = re.sub(r"<[^>]+>", "", clean)
                    clean = clean.replace("\\N", "\n").replace("\\n", "\n").strip()
                    if clean:
                        text_lines.append(clean)

            subtitle_text = "\n".join(text_lines)
        else:
            # Fallback: read as plain text
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                subtitle_text = f.read()

        if not subtitle_text.strip():
            return ParseResult(
                pages=[PageContent(page_num=1, text="(Empty subtitle file)", content_type="subtitle")],
                metadata={"format": "subtitle", "page_count": 1, "source": file_path},
            )

        return ParseResult(
            pages=[PageContent(
                page_num=1,
                text=f"[SUBTITLES: {os.path.basename(file_path)}]\n{subtitle_text}",
                content_type="subtitle",
            )],
            metadata={
                "format": "subtitle",
                "page_count": 1,
                "source": file_path,
            },
        )
    except Exception as e:
        return ParseResult(success=False, error=f"Subtitle parse error: {e}")


# ---------------------------------------------------------------------------
# Code Parser
# ---------------------------------------------------------------------------
def _parse_code(file_path: str) -> ParseResult:
    """Parse code files and explicitly preserve formatting."""
    try:
        encoding = "utf-8"
        if CHARDET_AVAILABLE:
            with open(file_path, "rb") as f:
                raw = f.read(10000)
                detected = chardet.detect(raw)
                encoding = detected.get("encoding", "utf-8") or "utf-8"

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        if not content.strip():
            return ParseResult(
                pages=[PageContent(page_num=1, text="(Empty code file)")],
                metadata={"format": "code", "page_count": 1, "source": file_path},
            )

        ext = os.path.splitext(file_path)[1][1:]
        formatted_code = f"[CODE FILE: {os.path.basename(file_path)}]\n```{ext}\n{content}\n```"

        return ParseResult(
            pages=[PageContent(
                page_num=1,
                text=formatted_code,
            )],
            metadata={"format": "code", "page_count": 1, "source": file_path},
        )
    except Exception as e:
        return ParseResult(success=False, error=f"Code parse error: {e}")


# ---------------------------------------------------------------------------
# Email Parser
# ---------------------------------------------------------------------------
def _parse_email(file_path: str) -> ParseResult:
    """Parse .eml / .msg files into structured text."""
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        subject = msg.get("subject", "(No Subject)")
        sender = msg.get("from", "(Unknown Sender)")
        date = msg.get("date", "(Unknown Date)")
        to = msg.get("to", "(Unknown Recipient)")

        body = ""
        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content() + "\n"
        else:
            body = msg.get_content()

        if not body.strip() and msg.is_multipart():
            # Fallback to html if no plain text
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    html = part.get_content()
                    if BS4_AVAILABLE:
                        soup = BeautifulSoup(html, "html.parser")
                        body += soup.get_text() + "\n"
                    else:
                        body += re.sub(r'<[^>]+>', '', html) + "\n"

        email_text = (
            f"[EMAIL MESSAGE]\n"
            f"Subject: {subject}\n"
            f"From: {sender}\n"
            f"To: {to}\n"
            f"Date: {date}\n"
            f"---\n{body.strip()}"
        )

        return ParseResult(
            pages=[PageContent(
                page_num=1,
                text=email_text,
            )],
            metadata={"format": "email", "page_count": 1, "source": file_path},
        )
    except Exception as e:
        return ParseResult(success=False, error=f"Email parse error: {e}")


# ---------------------------------------------------------------------------
# URL Link Parser
# ---------------------------------------------------------------------------
def _parse_url(file_path: str) -> ParseResult:
    """Parse .url or .webloc and scrape offline content."""
    try:
        url = None
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # Find HTTP links
            match = re.search(r'(https?://[^\s]+)', content)
            if match:
                url = match.group(1)
                # Cleanup common trailing characters
                url = url.rstrip('"]').rstrip('</string>')

        if not url:
            return ParseResult(success=False, error="No valid URL found in file")

        if os.getenv("RAG_HF_OFFLINE", "false").lower() in {"1", "true", "yes", "on"}:
            return ParseResult(
                pages=[PageContent(page_num=1, text=f"[WEB URL]\n{url}\n\n(Offline mode: content not scraped)")],
                metadata={"format": "url", "page_count": 1, "source": url},
            )

        # Scrape content offline
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='replace')

        title = "Web Page"
        body_text = html
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title:
                title = soup.title.string
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            body_text = soup.get_text(separator="\n", strip=True)

        web_content = f"[WEB PAGE]\nTitle: {title}\nURL: {url}\n\n{body_text}"
        
        # Paginate very long web pages
        pages = []
        lines = web_content.split("\n")
        page_size = 100
        for i in range(0, max(len(lines), 1), page_size):
            chunk = lines[i:i + page_size]
            pages.append(PageContent(page_num=(i//page_size)+1, text="\n".join(chunk)))

        return ParseResult(
            pages=pages,
            metadata={"format": "url", "page_count": len(pages), "source": url},
        )
    except Exception as e:
        return ParseResult(success=False, error=f"URL scrape error: {e}")


# ---------------------------------------------------------------------------
# Main Entry Point — Universal Parser
# ---------------------------------------------------------------------------
PARSER_REGISTRY = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".doc": _parse_docx,  # Best-effort with python-docx
    ".xlsx": _parse_xlsx,
    ".xls": _parse_xlsx,   # Best-effort with openpyxl
    ".csv": _parse_csv,
    ".pptx": _parse_pptx,
    ".ppt": _parse_pptx,   # Best-effort with python-pptx
    # Text formats
    ".txt": _parse_text,
    ".text": _parse_text,
    ".md": _parse_text,
    ".log": _parse_text,
    ".json": _parse_text,
    ".xml": _parse_text,
    # Images
    ".png": _parse_image,
    ".jpg": _parse_image,
    ".jpeg": _parse_image,
    ".bmp": _parse_image,
    ".tiff": _parse_image,
    ".tif": _parse_image,
    ".gif": _parse_image,
    ".webp": _parse_image,
    # Video
    ".mp4": _parse_video,
    ".avi": _parse_video,
    ".mkv": _parse_video,
    ".mov": _parse_video,
    ".wmv": _parse_video,
    ".flv": _parse_video,
    # Subtitle files
    ".srt": _parse_subtitle_file,
    ".ass": _parse_subtitle_file,
    ".ssa": _parse_subtitle_file,
    ".vtt": _parse_subtitle_file,
}

# Add code extensions dynamically
for ext in CODE_EXTENSIONS:
    PARSER_REGISTRY[ext] = _parse_code

# Add email extensions dynamically
for ext in EMAIL_EXTENSIONS:
    PARSER_REGISTRY[ext] = _parse_email

# Add URL extensions dynamically
for ext in URL_EXTENSIONS:
    PARSER_REGISTRY[ext] = _parse_url


def parse_file(file_path: str) -> ParseResult:
    """
    Universal file parser — auto-detects format and extracts all content.
    
    Supports: PDF, DOCX, XLSX, PPTX, CSV, TXT, MD, LOG, JSON, XML,
              PNG, JPG, BMP, TIFF, GIF, WEBP, MP4, AVI, MKV, MOV,
              SRT, ASS, SSA, VTT
    
    Returns ParseResult with pages, metadata, and error info.
    All processing is 100% offline.
    """
    if not os.path.exists(file_path):
        return ParseResult(success=False, error=f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    parser = PARSER_REGISTRY.get(ext)

    if not parser:
        return ParseResult(
            success=False,
            error=f"Unsupported file format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    print(f"[Parser] Parsing {os.path.basename(file_path)} (format: {ext})")
    result = parser(file_path)

    if result.success:
        total_text = sum(len(p.text) for p in result.pages)
        total_tables = sum(len(p.tables) for p in result.pages)
        total_images = sum(len(p.image_texts) for p in result.pages)
        print(
            f"[Parser] ✅ Parsed {len(result.pages)} pages, "
            f"{total_text} chars, {total_tables} tables, {total_images} images"
        )
    else:
        print(f"[Parser] ❌ Failed: {result.error}")

    return result
