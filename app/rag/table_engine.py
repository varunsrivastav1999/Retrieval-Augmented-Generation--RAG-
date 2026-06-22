"""
=============================================================================
 Enterprise Level RAG: Table Reconstruction Engine (v5.0)
=============================================================================
 Handles:
   - Merged cells (rowspan / colspan) via pdfplumber bounding-box geometry
   - Multi-level / nested header reconstruction
   - Multi-page table continuation stitching
   - Natural-language row serialization for table-aware embeddings
   - HTML table rendering for LLM prompts
   - Context-rich, 1-row-per-chunk chunking with full header paths

 This module is called by parsers.py (parse step) and ingestion.py
 (chunking step).  It never touches the DB or embedding models.
=============================================================================
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TableCell:
    """A single cell in a reconstructed table."""
    text: str = ""
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    resolved_header_path: str = ""  # e.g. "EQL > Single Phase > Dimensions > W"


@dataclass
class RichTable:
    """A fully reconstructed table with resolved spans and header paths."""
    table_id: str = ""
    section_title: str = ""          # The document section that owns this table
    page_range: List[int] = field(default_factory=list)  # [start_page, end_page]
    header_rows: List[List[TableCell]] = field(default_factory=list)
    data_rows: List[List[TableCell]] = field(default_factory=list)
    raw_headers: List[str] = field(default_factory=list)   # flat column names
    resolved_headers: List[str] = field(default_factory=list)  # full paths

    def to_markdown(self) -> str:
        """Export as GitHub-style markdown table."""
        if not self.resolved_headers and not self.data_rows:
            return ""
        headers = self.resolved_headers or self.raw_headers
        sep = " | ".join(["---"] * len(headers))
        lines = ["| " + " | ".join(headers) + " |", "| " + sep + " |"]
        for row in self.data_rows:
            cells = [c.text.replace("|", "\\|").replace("\n", " ") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def to_html(self) -> str:
        """Export as clean HTML for LLM prompts (better column alignment)."""
        if not self.data_rows:
            return ""
        lines = [f"<table>"]
        if self.section_title:
            lines.insert(0, f"<caption>{self.section_title}</caption>")
        # thead
        lines.append("<thead>")
        for hrow in self.header_rows:
            lines.append("<tr>" + "".join(
                f"<th colspan='{c.colspan}' rowspan='{c.rowspan}'>{c.text}</th>"
                for c in hrow
            ) + "</tr>")
        if not self.header_rows and self.resolved_headers:
            lines.append("<tr>" + "".join(
                f"<th>{h}</th>" for h in self.resolved_headers
            ) + "</tr>")
        lines.append("</thead>")
        # tbody
        lines.append("<tbody>")
        for row in self.data_rows:
            lines.append("<tr>" + "".join(
                f"<td>{c.text}</td>" for c in row
            ) + "</tr>")
        lines.append("</tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    def row_to_dict(self, row: List[TableCell]) -> Dict[str, str]:
        """Return a {header: value} dict for a data row."""
        headers = self.resolved_headers or self.raw_headers
        return {
            (headers[i] if i < len(headers) else f"col_{i}"): cell.text
            for i, cell in enumerate(row)
        }

    def row_to_natural_language(self, row: List[TableCell]) -> str:
        """
        Serialize a table row as a natural-language sentence.
        Example: "24-circuit EQL Single Phase loadcentre (catalogue ECL2412SD)
                  has dimensions W=13⁵⁄₁₆ in, H=34 in, D=5½ in, Lug Data=4-2/0 MCM,
                  Mounting=combo, Door Kit=DK10-1A."
        """
        headers = self.resolved_headers or self.raw_headers
        parts = []
        for i, cell in enumerate(row):
            if not cell.text.strip():
                continue
            hdr = headers[i] if i < len(headers) else f"col_{i}"
            parts.append(f"{hdr}: {cell.text.strip()}")
        section = f"[{self.section_title}] " if self.section_title else ""
        return section + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Span Matrix Builder — resolves rowspan / colspan from pdfplumber geometry
# ---------------------------------------------------------------------------

def _build_span_matrix_from_raw(
    raw_rows: List[List[str]],
    bbox_rows: Optional[List[List[Tuple]]] = None,
) -> Tuple[List[List[TableCell]], List[int]]:
    """
    Convert raw cell text (and optional bbox geometry) into a full SpanMatrix.
    Detects header rows heuristically (short rows, ALL CAPS, bold-only content).
    Returns (grid, header_row_indices).
    """
    grid: List[List[TableCell]] = []
    header_indices: List[int] = []

    for r_idx, raw_row in enumerate(raw_rows):
        row_cells: List[TableCell] = []
        for c_idx, cell_text in enumerate(raw_row):
            text = str(cell_text or "").strip()
            cell = TableCell(
                text=text,
                row=r_idx,
                col=c_idx,
            )
            row_cells.append(cell)
        grid.append(row_cells)

    # Heuristic: first rows that look like headers
    # (short text, uppercase, or appear before data rows with numbers)
    for r_idx, row in enumerate(grid):
        texts = [c.text for c in row if c.text]
        if not texts:
            continue
        is_all_caps_or_short = all(
            len(t) <= 30 or t.isupper() or not any(ch.isdigit() for ch in t)
            for t in texts
        )
        has_numeric_row_below = any(
            any(ch.isdigit() for ch in c.text)
            for c in (grid[r_idx + 1] if r_idx + 1 < len(grid) else [])
        )
        if is_all_caps_or_short and has_numeric_row_below and r_idx < 4:
            header_indices.append(r_idx)
            for c in row:
                c.is_header = True

    return grid, header_indices


def _resolve_nested_headers(grid: List[List[TableCell]], header_indices: List[int]) -> List[str]:
    """
    Walk multi-row headers and build a "Level1 > Level2 > ColumnName" path
    for each column, then return a flat list of resolved column names.

    Example input (2 header rows, 8 cols):
      Row 0: ["Number of Circuits", "Catalogue Number", "Slot Qty", "Main Amps",
               "Dimensions (Inches/mm)", "", "", "Lug Data", ...]
      Row 1: ["", "", "", "", "H", "W", "D", "", ...]

    Output: ["Number of Circuits", "Catalogue Number", "Slot Qty", "Main Amps",
             "Dimensions > H", "Dimensions > W", "Dimensions > D", "Lug Data", ...]
    """
    if not header_indices:
        # Use first row as fallback headers
        if grid:
            return [c.text or f"col_{i}" for i, c in enumerate(grid[0])]
        return []

    # Determine column count
    col_count = max(len(grid[i]) for i in header_indices) if header_indices else 0
    if col_count == 0:
        return []

    # Build column-wise header path
    column_paths: List[List[str]] = [[] for _ in range(col_count)]

    for r_idx in header_indices:
        row = grid[r_idx]
        last_non_empty = ""
        for c_idx in range(col_count):
            cell_text = row[c_idx].text if c_idx < len(row) else ""
            if cell_text:
                last_non_empty = cell_text
                column_paths[c_idx].append(cell_text)
            else:
                # Inherit spanning header from the left
                column_paths[c_idx].append(last_non_empty)

    resolved = []
    for path in column_paths:
        # Remove consecutive duplicates, then join with " > "
        deduped = []
        for p in path:
            if p and (not deduped or deduped[-1] != p):
                deduped.append(p)
        resolved.append(" > ".join(deduped) if deduped else "")

    return resolved


# ---------------------------------------------------------------------------
# pdfplumber geometry-aware table extractor
# ---------------------------------------------------------------------------

def extract_tables_pdfplumber(file_path: str, page_no: int) -> List[RichTable]:
    """
    Extract tables from a specific page using pdfplumber with explicit
    line detection for merged cell geometry.
    Returns a list of RichTable objects (may be empty if no tables found).
    """
    if not PDFPLUMBER_AVAILABLE:
        return []

    tables: List[RichTable] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            if page_no < 1 or page_no > len(pdf.pages):
                return []
            page = pdf.pages[page_no - 1]

            # Strategy 1: lines-based (best for bordered tables)
            table_settings_lines = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 3,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
                "text_tolerance": 3,
                "text_x_tolerance": 3,
                "text_y_tolerance": 3,
            }
            raw_tables = page.extract_tables(table_settings=table_settings_lines)

            # Strategy 2: text-based fallback (for borderless / ruled tables)
            if not raw_tables:
                table_settings_text = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                }
                raw_tables = page.extract_tables(table_settings=table_settings_text)

            for t_idx, raw_table in enumerate(raw_tables or []):
                if not raw_table:
                    continue
                grid, header_indices = _build_span_matrix_from_raw(raw_table)
                resolved_headers = _resolve_nested_headers(grid, header_indices)

                # Separate header rows from data rows
                data_start = (max(header_indices) + 1) if header_indices else 1
                header_cell_rows = [grid[i] for i in header_indices]
                data_cell_rows = grid[data_start:]

                # Apply bi-directional empty-cell inheritance on data rows
                data_cell_rows = _fill_empty_cells(data_cell_rows)

                rich = RichTable(
                    table_id=f"p{page_no}_t{t_idx}",
                    page_range=[page_no],
                    header_rows=header_cell_rows,
                    data_rows=data_cell_rows,
                    raw_headers=[c.text for c in (grid[0] if grid else [])],
                    resolved_headers=resolved_headers,
                )
                tables.append(rich)
    except Exception as e:
        print(f"[TableEngine] pdfplumber extraction failed page {page_no}: {e}")

    return tables


def _fill_empty_cells(rows: List[List[TableCell]]) -> List[List[TableCell]]:
    """
    Bi-directional empty cell inheritance.
    Forward fill (top-to-bottom) then backward fill (bottom-to-top)
    to reconstruct merged cells that were flattened.
    """
    if not rows:
        return rows

    col_count = max(len(r) for r in rows)

    # Pad rows to same length
    for row in rows:
        while len(row) < col_count:
            row.append(TableCell())

    # Forward fill (top → bottom)
    for r_idx in range(1, len(rows)):
        for c_idx in range(col_count):
            if not rows[r_idx][c_idx].text and rows[r_idx - 1][c_idx].text:
                rows[r_idx][c_idx].text = rows[r_idx - 1][c_idx].text

    # Backward fill (bottom → top)
    for r_idx in range(len(rows) - 2, -1, -1):
        for c_idx in range(col_count):
            if not rows[r_idx][c_idx].text and rows[r_idx + 1][c_idx].text:
                rows[r_idx][c_idx].text = rows[r_idx + 1][c_idx].text

    return rows


# ---------------------------------------------------------------------------
# Multi-page table stitcher
# ---------------------------------------------------------------------------

def stitch_continuation_tables(tables_by_page: Dict[int, List[RichTable]]) -> Dict[int, List[RichTable]]:
    """
    Detect and stitch tables that continue across page boundaries.
    Heuristic: if page N has a table ending near the bottom and page N+1
    starts a table with the same column count (±1) and no header row,
    they are merged.

    Returns the same dict structure but with stitched tables merged into
    the first page's entry, and removed from subsequent pages.
    """
    page_nums = sorted(tables_by_page.keys())
    stitched_pages: set = set()

    for i, page_no in enumerate(page_nums[:-1]):
        next_page = page_nums[i + 1]
        if next_page != page_no + 1:
            continue  # Non-consecutive pages

        page_tables = tables_by_page.get(page_no, [])
        next_tables = tables_by_page.get(next_page, [])

        if not page_tables or not next_tables:
            continue

        last_table = page_tables[-1]
        first_next = next_tables[0]

        if _is_continuation(last_table, first_next):
            print(f"[TableEngine] Stitching table from page {page_no} → {next_page}")
            # Merge data rows
            last_table.data_rows.extend(first_next.data_rows)
            last_table.page_range.append(next_page)

            # Remove the continuation table from the next page
            tables_by_page[next_page] = next_tables[1:]
            stitched_pages.add((page_no, next_page))

    return tables_by_page


def _is_continuation(table_a: RichTable, table_b: RichTable) -> bool:
    """
    Returns True if table_b is a continuation of table_a.
    Checks:
    1. Column count matches (±1 tolerance)
    2. table_b has no header rows (or headers identical to table_a)
    3. First data row of table_b is NOT a full header row (all text, no numbers)
    """
    col_a = len(table_a.resolved_headers) or len(table_a.raw_headers)
    col_b = len(table_b.resolved_headers) or len(table_b.raw_headers)

    if abs(col_a - col_b) > 1:
        return False  # Completely different column count

    # If table_b has its own full header block, it's a new table
    if table_b.header_rows:
        # Check if headers are nearly identical (same table)
        hdr_a = set(table_a.resolved_headers)
        hdr_b = set(table_b.resolved_headers)
        if hdr_a and hdr_b and len(hdr_a & hdr_b) / max(len(hdr_a), 1) > 0.7:
            # Same headers — it's a continuation with repeated header (common in PDFs)
            return True
        return False  # Different headers — new table

    # No header rows in table_b — likely a continuation
    if not table_b.data_rows:
        return False

    first_row = table_b.data_rows[0]
    texts = [c.text for c in first_row if c.text]
    # If first row contains only non-numeric text it may be a misdetected header
    # (some PDFs repeat section titles). Don't merge in that case.
    if texts and all(not any(ch.isdigit() for ch in t) for t in texts) and len(texts) == len(first_row):
        return False

    return True


# ---------------------------------------------------------------------------
# Section title annotator
# ---------------------------------------------------------------------------

def annotate_section_title(surrounding_text: str, fallback: str = "") -> str:
    """
    Extract the most likely section title for a table from surrounding text.
    Looks for lines that appear to be headings (ALL CAPS, short, or followed
    by 'Selection and Ordering Data' / 'Technical Data').
    """
    if not surrounding_text:
        return fallback

    lines = surrounding_text.strip().split("\n")
    candidates = []

    heading_patterns = [
        r"^[A-Z][A-Za-z0-9& ,\-/]{3,60}$",          # Title Case / Sentence Case
        r"^[A-Z ]{4,50}$",                            # ALL CAPS heading
        r"^(EQL|SNC|SEQ|EQ3|EQ4|SQD|ECL|SNC)\b",    # Product families
    ]

    for line in reversed(lines):  # Search from bottom-up (closest to table)
        line = line.strip()
        if not line or len(line) > 80:
            continue
        for pat in heading_patterns:
            if re.match(pat, line):
                candidates.append(line)
                break

    return candidates[0] if candidates else fallback


# ---------------------------------------------------------------------------
# Context-rich chunker  (replaces ingestion.chunk_table_per_row)
# ---------------------------------------------------------------------------

def chunk_rich_table(
    table: RichTable,
    doc_title: str,
    max_adjacent_rows: int = 1,
) -> List[Dict]:
    """
    Chunk a RichTable into 1-row-per-chunk entries.

    Each chunk dict contains:
      text          — full markdown block (header + ±1 adjacent rows)
      nl_text       — natural-language sentence (used as primary embedding)
      json_cells    — {header: value} dict for exact-lookup metadata
      header_path   — resolved column headers list
      section_title — owning section
      table_group   — stable table ID
      table_id      — same as table_group (compatibility alias)
      page_range    — [first_page, last_page]
      row_index     — row ordinal within the table
      content_type  — "table_row"
    """
    chunks = []
    if not table.data_rows:
        return chunks

    headers = table.resolved_headers or table.raw_headers
    n_rows = len(table.data_rows)

    for r_idx, row in enumerate(table.data_rows):
        # Skip empty rows
        row_texts = [c.text for c in row if c.text.strip()]
        if not row_texts:
            continue

        # Build adjacent context window (±max_adjacent_rows)
        ctx_start = max(0, r_idx - max_adjacent_rows)
        ctx_end = min(n_rows, r_idx + max_adjacent_rows + 1)
        ctx_rows = table.data_rows[ctx_start:ctx_end]

        # Markdown block: headers + adjacent rows (for retrieval scoring)
        md_header = "| " + " | ".join(headers) + " |"
        md_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        md_rows = []
        for cr in ctx_rows:
            cells = [c.text.replace("|", "\\|").replace("\n", " ") for c in cr]
            while len(cells) < len(headers):
                cells.append("")
            md_rows.append("| " + " | ".join(cells) + " |")

        section_label = f"[{table.section_title}] " if table.section_title else ""
        text = (
            f"[{doc_title}] {section_label}\n"
            f"{md_header}\n{md_sep}\n"
            + "\n".join(md_rows)
        )

        # Natural-language sentence (primary embedding text)
        nl_text = table.row_to_natural_language(row)

        # JSON cell dict for exact-match filtering
        json_cells = table.row_to_dict(row)

        chunks.append({
            "text": text,
            "nl_text": nl_text,
            "json_cells": json_cells,
            "header_path": headers,
            "section_title": table.section_title,
            "table_group": table.table_id,
            "table_id": table.table_id,
            "page_range": table.page_range,
            "row_index": r_idx,
            "content_type": "table_row",
            "is_parent": True,
            "parent_idx": None,
            "child_idx": None,
        })

    return chunks


# ---------------------------------------------------------------------------
# Markdown table → RichTable (for Docling output)
# ---------------------------------------------------------------------------

def markdown_to_rich_table(
    markdown_table: str,
    table_id: str = "",
    section_title: str = "",
    page_no: int = 1,
) -> Optional[RichTable]:
    """
    Convert a Docling-produced markdown table string into a RichTable.
    Preserves as much structure as possible via the nested-header resolver.
    This path is used when pdfplumber geometry is unavailable.
    """
    lines = [ln for ln in markdown_table.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return None

    # Find separator
    sep_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and re.match(r"^[\|\-\: ]+$", stripped):
            sep_idx = i
            break

    if sep_idx == -1:
        return None

    def _parse_md_row(line: str) -> List[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.strip() for c in line.split("|")]

    # Everything before separator = potential header rows
    raw_headers_rows = [_parse_md_row(lines[i]) for i in range(sep_idx)]
    raw_data_rows = [_parse_md_row(lines[i]) for i in range(sep_idx + 1, len(lines))]

    # Build header grid
    header_grid = []
    for r in raw_headers_rows:
        header_grid.append([TableCell(text=t, is_header=True) for t in r])

    header_indices = list(range(len(raw_headers_rows)))
    resolved_headers = _resolve_nested_headers(header_grid, header_indices)

    # Build data grid with empty-cell fill
    data_grid = []
    for r in raw_data_rows:
        data_grid.append([TableCell(text=t) for t in r])
    data_grid = _fill_empty_cells(data_grid)

    return RichTable(
        table_id=table_id or f"p{page_no}_md",
        section_title=section_title,
        page_range=[page_no],
        header_rows=header_grid,
        data_rows=data_grid,
        raw_headers=raw_headers_rows[0] if raw_headers_rows else [],
        resolved_headers=resolved_headers,
    )


# ---------------------------------------------------------------------------
# HTML table assembler  (called by context.py)
# ---------------------------------------------------------------------------

def assemble_html_table_from_chunks(chunks: List[Dict]) -> str:
    """
    Given a list of table_row chunk dicts (from retrieval),
    reconstruct a clean HTML table for the LLM prompt.
    Groups by table_id, sorts by row_index, renders as HTML.
    """
    from collections import defaultdict

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        tid = meta.get("table_id") or meta.get("table_group") or "unknown"
        grouped[tid].append(chunk)

    html_parts = []
    for tid, rows in grouped.items():
        rows.sort(key=lambda r: r.get("metadata", {}).get("row_index", 0))

        section = rows[0].get("metadata", {}).get("section_title", "") if rows else ""
        headers = rows[0].get("metadata", {}).get("header_path", []) if rows else []

        lines = []
        if section:
            lines.append(f"<caption><strong>{section}</strong></caption>")
        lines.append("<table border='1' cellspacing='0' cellpadding='4'>")
        if headers:
            lines.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
        lines.append("<tbody>")
        for row in rows:
            cells = row.get("metadata", {}).get("cell_values", {})
            if cells:
                lines.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells.values()) + "</tr>")
            else:
                # Fallback: extract from text
                text = row.get("text", "")
                lines.append(f"<tr><td colspan='{max(len(headers),1)}'>{text[:300]}</td></tr>")
        lines.append("</tbody></table>")
        html_parts.append("\n".join(lines))

    return "\n\n".join(html_parts)


# ---------------------------------------------------------------------------
# Query classifier  (called by retrieval.py / main.py)
# ---------------------------------------------------------------------------

class QueryType:
    TEXT_RETRIEVAL = "text"
    TABLE_LOOKUP   = "table_lookup"
    TABLE_COMPARE  = "table_compare"
    TABLE_AGGREGATE = "table_aggregate"
    CALCULATION    = "calculation"


_LOOKUP_SIGNALS    = ["catalogue number", "model number", "part number", "what is the",
                      "dimensions of", "rating of", "door kit", "factory modification",
                      "ordering data", "lug data", "mounting"]
_COMPARE_SIGNALS   = ["compare", "difference between", " vs ", "versus", "better than",
                      "which is larger", "which has more"]
_AGGREGATE_SIGNALS = ["list all", "all models", "how many", "which models", "show all",
                      "give me all", "every model", "all circuits"]
_CALC_SIGNALS      = ["total", "sum", "average", "calculate", "how much does"]


def classify_query(query: str) -> str:
    """
    Classify a query into one of the QueryType constants.
    Fast rule-based classifier — no LLM call.
    """
    q = query.lower()
    if any(s in q for s in _COMPARE_SIGNALS):
        return QueryType.TABLE_COMPARE
    if any(s in q for s in _AGGREGATE_SIGNALS):
        return QueryType.TABLE_AGGREGATE
    if any(s in q for s in _CALC_SIGNALS):
        return QueryType.CALCULATION
    if any(s in q for s in _LOOKUP_SIGNALS):
        return QueryType.TABLE_LOOKUP
    # Pattern: contains a catalogue-number-like token
    if re.search(r'\b[A-Z]{2,5}\d{2,6}[A-Z0-9\-]*\b', query):
        return QueryType.TABLE_LOOKUP
    return QueryType.TEXT_RETRIEVAL


def extract_catalogue_patterns(query: str) -> List[str]:
    """Extract potential catalogue/model number patterns from a query string."""
    return re.findall(r'\b[A-Z]{2,5}\d{2,6}[A-Z0-9\-]*\b', query)
