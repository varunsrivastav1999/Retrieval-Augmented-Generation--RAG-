"""Section-aware chunking.

* Text is split at headings into sections, then into ~RAG_CHUNK_CHARS chunks on
  paragraph boundaries (never mid-line). Each chunk keeps its section id so the
  whole section (e.g. a complete alarm entry or procedure) can be reassembled
  in order at answer time — no duplicated "parent text" storage.
* The embedding text is prefixed with "document | section" so every chunk of
  an alarm entry carries the alarm code even if the body never repeats it.
* Table rows and record rows become one chunk each ("Header: value; ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

from app.rag.identifiers import extract_identifiers, normalize_identifier

_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
# "3.2 Bell cup cleaning", "4.1.2 Alarm list", "A.1 Wiring" (multi-level only: "1. Remove cover" is a step)
_NUM_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)+\.?|[A-Z]\.\d+(?:\.\d+)*)\s+([A-Z].{2,90})$")
# "3 Maintenance" (single level, no dot, short)
_NUM1_HEADING = re.compile(r"^\s*\d{1,2}\s+[A-Z][A-Za-z/&,()\- ]{2,60}$")
_SAFETY = re.compile(r"^(WARNING|CAUTION|DANGER|NOTE|NOTICE|IMPORTANT|ATTENTION)\b")
_KEYWORD_HEADING = re.compile(r"^\s*(alarm|error|fault|chapter|section|procedure|appendix)\b.{2,100}$", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class ChunkDraft:
    text: str
    embed_text: str
    content_type: str = "text"          # text | table_row | record
    page: Optional[int] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    identifiers: List[str] = field(default_factory=list)


def _heading(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    m = _MD_HEADING.match(stripped)
    if m:
        return m.group(2).strip()
    if _SAFETY.match(stripped.upper()):
        return None
    if _NUM_HEADING.match(stripped) and not stripped.endswith((".", ",", ";", ":")) and len(stripped.split()) <= 14:
        return stripped
    if _NUM1_HEADING.match(stripped) and len(stripped.split()) <= 7:
        return stripped
    if _KEYWORD_HEADING.match(stripped) and not stripped.endswith((".", ",", ";")) and len(stripped.split()) <= 14:
        return stripped
    if stripped.isupper() and 2 <= len(stripped.split()) <= 10 and re.search(r"[A-Z]{3}", stripped):
        return stripped
    return None


def _ids(text: str) -> List[str]:
    return list(dict.fromkeys(normalize_identifier(i) for i in extract_identifiers(text, limit=64)))


def _doc_ids(doc_name: str) -> List[str]:
    """Identifiers in the file name (e.g. machine model "P-500" in p-500_maintenance_manual.pdf)."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", doc_name)
    return _ids(re.sub(r"[_.]+", " ", stem).upper())


def _make(doc_name: str, body: str, *, page, section_id, section_title, content_type="text") -> ChunkDraft:
    context = f"{doc_name} | {section_title}" if section_title else doc_name
    embed_text = f"{context}\n{body}"
    ids = list(dict.fromkeys(_ids(embed_text) + _doc_ids(doc_name)))
    return ChunkDraft(text=body, embed_text=embed_text, content_type=content_type, page=page,
                      section_id=section_id, section_title=section_title, identifiers=ids)


def _split_long(paragraph: str, max_chars: int) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out, buf = [], ""
    for sent in _SENTENCE_SPLIT.split(paragraph):
        while len(sent) > max_chars:  # pathological: no punctuation at all
            out.append(sent[:max_chars])
            sent = sent[max_chars:]
        if buf and len(buf) + len(sent) + 1 > max_chars:
            out.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return out


def _flush(doc_name: str, buf: List[str], page: Optional[int], section_id, section_title) -> Optional[ChunkDraft]:
    body = "\n".join(buf).strip()
    # Short bodies are noise, unless they belong to a titled section (e.g. a one-line alarm fix).
    if len(body) < 20 and not (section_title and len(body) >= 3):
        return None
    return _make(doc_name, body, page=page, section_id=section_id, section_title=section_title)


def chunk_pages(doc_name: str, pages: Sequence, section_prefix: str, max_chars: int = 900) -> Iterator[ChunkDraft]:
    """Chunk parsed pages (parsers.PageContent) into ChunkDrafts."""
    section_no = 0
    section_title: Optional[str] = None
    section_id: Optional[str] = f"{section_prefix}-0"

    for page in pages:
        buf: List[str] = []
        size = 0
        section_at_page_start = section_title

        for raw_line in (page.text or "").split("\n"):
            title = _heading(raw_line)
            if title:
                chunk = _flush(doc_name, buf, page.page_num, section_id, section_title)
                buf, size = [], 0
                if chunk:
                    yield chunk
                section_no += 1
                section_title = title[:300]
                section_id = f"{section_prefix}-{section_no}"
                continue
            line = raw_line.strip()
            if not line:
                if buf and buf[-1] != "":
                    buf.append("")
                continue
            for piece in _split_long(line, max_chars):
                if size and size + len(piece) > max_chars:
                    chunk = _flush(doc_name, buf, page.page_num, section_id, section_title)
                    buf, size = [], 0
                    if chunk:
                        yield chunk
                buf.append(piece)
                size += len(piece) + 1
        chunk = _flush(doc_name, buf, page.page_num, section_id, section_title)
        if chunk:
            yield chunk

        # Tables carry their own heading when the parser found one above them; otherwise
        # they belong to the section that was open when the page started.
        for table in getattr(page, "rich_tables", []) or []:
            yield from _table_rows(doc_name, table, page.page_num, section_at_page_start)
        for md in getattr(page, "tables", []) or []:
            yield from _markdown_table_rows(doc_name, md, page.page_num, section_at_page_start)


def _row_text(header: Sequence[str], row: Sequence[str]) -> str:
    pairs = [f"{h}: {v}" for h, v in zip(header, row) if v not in ("", None)]
    extra = [v for v in row[len(header):] if v]
    return "; ".join(pairs + extra)


def _table_rows(doc_name, table, page_num, section_title) -> Iterator[ChunkDraft]:
    headers = table.resolved_headers or table.raw_headers
    caption = table.section_title or section_title
    for row in table.data_rows:
        body = _row_text(headers, [c.text.strip() for c in row])
        if len(body) >= 5:
            yield _make(doc_name, body, page=page_num, section_id=None, section_title=caption, content_type="table_row")


def _markdown_table_rows(doc_name, md: str, page_num, section_title) -> Iterator[ChunkDraft]:
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return
    split = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]  # noqa: E731
    header = split(lines[0])
    for line in lines[2:]:
        body = _row_text(header, split(line))
        if len(body) >= 5:
            yield _make(doc_name, body, page=page_num, section_id=None, section_title=section_title, content_type="table_row")


def record_chunk(doc_name: str, sheet: str, header: Sequence[str], row: Sequence[str], row_number: int) -> Optional[ChunkDraft]:
    body = _row_text(header, row)
    if len(body) < 3:
        return None
    return _make(doc_name, body, page=row_number, section_id=None, section_title=sheet or None, content_type="record")
