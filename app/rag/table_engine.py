"""Table reconstruction helpers (offline, pure Python).

* markdown -> RichTable with nested-header resolution and merged-cell filling;
* stitching of tables that continue across page breaks.

Used by parsers.py; chunking.py turns each data row into one chunk
("Header: value; ...") so exact values stay retrievable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

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



def _fill_empty_cells(rows: List[List[TableCell]]) -> List[List[TableCell]]:
    """
    Bi-directional empty cell inheritance.
    Forward fill (top-to-bottom) then backward fill (bottom-to-top)
    to reconstruct merged cells that were flattened.

    Uses column fill-rate heuristic: only fills columns where < 50% of cells
    have data (indicating they are likely merged/spanning cells).
    Columns with > 50% fill are treated as having their own data per-row.
    """
    if not rows:
        return rows

    col_count = max(len(r) for r in rows)

    # Pad rows to same length
    for row in rows:
        while len(row) < col_count:
            row.append(TableCell())

    # Calculate fill rate per column to identify truly merged columns
    n_rows = len(rows)
    col_fill_rate = []
    for c_idx in range(col_count):
        filled = sum(1 for r in rows if r[c_idx].text.strip())
        col_fill_rate.append(filled / max(n_rows, 1))

    # Forward fill (top → bottom) — only for columns with low fill rate (likely merged)
    for r_idx in range(1, len(rows)):
        for c_idx in range(col_count):
            if col_fill_rate[c_idx] > 0.5:
                continue  # This column has its own data per-row, don't fill
            if not rows[r_idx][c_idx].text and rows[r_idx - 1][c_idx].text:
                rows[r_idx][c_idx].text = rows[r_idx - 1][c_idx].text

    # Backward fill (bottom → top) — only for columns with low fill rate
    for r_idx in range(len(rows) - 2, -1, -1):
        for c_idx in range(col_count):
            if col_fill_rate[c_idx] > 0.5:
                continue  # This column has its own data per-row, don't fill
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
            log.debug("stitching table from page %s to %s", page_no, next_page)
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
    1. Section titles must match (if both present)
    2. Column count matches (±1 tolerance)
    3. table_b has no header rows (or headers identical to table_a)
    4. First data row of table_b is NOT a full header row (all text, no numbers)
    """
    # If both tables have explicit section titles and they differ, it's a new table
    if (table_a.section_title and table_b.section_title
            and table_a.section_title.strip() != table_b.section_title.strip()):
        return False

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
