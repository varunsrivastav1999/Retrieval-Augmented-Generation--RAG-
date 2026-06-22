"""
=============================================================================
 Enterprise Level RAG: Canonical Table Store v1.0
=============================================================================
 Stores fully-reconstructed table rows as structured JSON in PostgreSQL.
 Enables:
   - 0-token exact lookup via SQL GIN index on cells JSONB
   - 0-token aggregation via SQL GROUP BY / COUNT / SUM
   - Range queries via numeric_cells JSONB (e.g. amps >= 200)
   - Cross-table comparison via JOIN on section_title
   - Note reference resolution via note_refs JSONB

 This module is called by ingestion.py during document processing.
 Query execution is called by retrieval.py for EXACT_LOOKUP / AGGREGATION routes.
=============================================================================
"""

from __future__ import annotations

import json
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from sqlalchemy import text
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CanonicalTableRow:
    """
    One row of a fully-reconstructed table.
    Maps directly to the canonical_table_rows PostgreSQL table.
    """
    tenant_id: str
    doc_id: str
    table_id: str
    section_title: str
    page_start: int
    page_end: int
    row_index: int
    header_path: Dict[str, str]          # {col_0: "Catalogue Number", col_1: "No. of Circuits"}
    cells: Dict[str, str]                # {"Catalogue Number": "ECL2412SD", ...}
    numeric_cells: Dict[str, float]      # {"amps": 200.0, "circuits": 24.0}
    nl_sentence: str                     # Embeddable natural-language sentence
    note_refs: Dict[str, str]            # {"①": "footnote text"}
    row_id: str = ""                     # Auto-computed hash

    def __post_init__(self):
        if not self.row_id:
            self.row_id = self._compute_row_id()

    def _compute_row_id(self) -> str:
        key = f"{self.doc_id}::{self.table_id}::{self.row_index}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_insert_dict(self) -> dict:
        return {
            "tenant_id":    self.tenant_id,
            "doc_id":       self.doc_id,
            "table_id":     self.table_id,
            "section_title": self.section_title,
            "page_start":   self.page_start,
            "page_end":     self.page_end,
            "row_id":       self.row_id,
            "row_index":    self.row_index,
            "header_path":  json.dumps(self.header_path),
            "cells":        json.dumps(self.cells),
            "numeric_cells": json.dumps(self.numeric_cells),
            "nl_sentence":  self.nl_sentence,
            "note_refs":    json.dumps(self.note_refs),
        }


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS canonical_table_rows (
    id            SERIAL PRIMARY KEY,
    tenant_id     VARCHAR NOT NULL,
    doc_id        TEXT NOT NULL,
    table_id      VARCHAR NOT NULL,
    section_title VARCHAR,
    page_start    INTEGER,
    page_end      INTEGER,
    row_id        VARCHAR UNIQUE NOT NULL,
    row_index     INTEGER,
    header_path   JSONB,
    cells         JSONB NOT NULL,
    numeric_cells JSONB,
    nl_sentence   TEXT,
    note_refs     JSONB,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ctr_tenant     ON canonical_table_rows(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_ctr_doc        ON canonical_table_rows(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_ctr_table_id   ON canonical_table_rows(table_id);",
    "CREATE INDEX IF NOT EXISTS idx_ctr_section    ON canonical_table_rows(section_title);",
    "CREATE INDEX IF NOT EXISTS idx_ctr_cells_gin  ON canonical_table_rows USING gin(cells);",
    "CREATE INDEX IF NOT EXISTS idx_ctr_num_gin    ON canonical_table_rows USING gin(numeric_cells);",
    "CREATE INDEX IF NOT EXISTS idx_ctr_row_id     ON canonical_table_rows(row_id);",
]


def init_canonical_store(db: Session) -> None:
    """Create the canonical_table_rows table and indexes if they don't exist."""
    try:
        db.execute(text(CREATE_TABLE_SQL))
        db.commit()
        for idx_sql in CREATE_INDEXES_SQL:
            try:
                db.execute(text(idx_sql))
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"[CanonicalStore] Index creation skipped: {e}")
        print("[CanonicalStore] Schema initialized.")
    except Exception as e:
        db.rollback()
        print(f"[CanonicalStore] Schema init failed (may already exist): {e}")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def upsert_canonical_rows(db: Session, rows: List[CanonicalTableRow]) -> int:
    """
    Upsert a list of CanonicalTableRow objects.
    Uses ON CONFLICT (row_id) DO UPDATE for idempotent re-ingestion.
    Returns number of rows inserted/updated.
    """
    if not rows:
        return 0

    sql = text("""
        INSERT INTO canonical_table_rows
            (tenant_id, doc_id, table_id, section_title, page_start, page_end,
             row_id, row_index, header_path, cells, numeric_cells, nl_sentence, note_refs)
        VALUES
            (:tenant_id, :doc_id, :table_id, :section_title, :page_start, :page_end,
             :row_id, :row_index, :header_path::jsonb, :cells::jsonb,
             :numeric_cells::jsonb, :nl_sentence, :note_refs::jsonb)
        ON CONFLICT (row_id) DO UPDATE SET
            cells         = EXCLUDED.cells,
            numeric_cells = EXCLUDED.numeric_cells,
            nl_sentence   = EXCLUDED.nl_sentence,
            section_title = EXCLUDED.section_title,
            note_refs     = EXCLUDED.note_refs
    """)

    count = 0
    for row in rows:
        try:
            db.execute(sql, row.to_insert_dict())
            count += 1
        except Exception as e:
            db.rollback()
            print(f"[CanonicalStore] Upsert failed for row_id={row.row_id}: {e}")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[CanonicalStore] Commit failed: {e}")

    return count


def delete_doc_rows(db: Session, tenant_id: str, doc_id: str) -> None:
    """Remove all canonical rows for a document (used during force re-index)."""
    try:
        db.execute(text(
            "DELETE FROM canonical_table_rows WHERE tenant_id=:t AND doc_id=:d"
        ), {"t": tenant_id, "d": doc_id})
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[CanonicalStore] Delete failed: {e}")


# ---------------------------------------------------------------------------
# Exact lookup (0 tokens, ~0ms)
# ---------------------------------------------------------------------------

def exact_lookup(
    db: Session,
    tenant_id: str,
    query: str,
    catalogue_patterns: List[str] = None,
    numeric_filters: Dict[str, Tuple[str, float]] = None,
    section_title_hint: str = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Exact table cell lookup using GIN indexes on cells JSONB and ILIKE text search.
    Returns a list of matching row dicts with full cell values.

    Priority:
    1. GIN @> match on catalogue patterns (fastest, index scan)
    2. ILIKE text match on cells::text (fallback)
    3. Numeric range filter on numeric_cells (combined with above)
    """
    results = []
    seen_row_ids: set = set()

    # ── Strategy 1: Direct catalogue pattern match via ILIKE on cells::text
    if catalogue_patterns:
        for pattern in catalogue_patterns[:3]:
            sql = text("""
                SELECT row_id, table_id, section_title, page_start, page_end,
                       row_index, header_path, cells, numeric_cells, nl_sentence, note_refs
                FROM canonical_table_rows
                WHERE tenant_id = :tenant_id
                  AND cells::text ILIKE :pattern
                  {section_filter}
                ORDER BY row_index
                LIMIT :limit
            """.format(
                section_filter="AND section_title ILIKE :section" if section_title_hint else ""
            ))
            params = {
                "tenant_id": tenant_id,
                "pattern": f"%{pattern}%",
                "limit": limit,
            }
            if section_title_hint:
                params["section"] = f"%{section_title_hint}%"

            try:
                rows = db.execute(sql, params).fetchall()
                for row in rows:
                    if row.row_id in seen_row_ids:
                        continue
                    # Apply numeric filter if present
                    if numeric_filters and not _passes_numeric_filter(row.numeric_cells or {}, numeric_filters):
                        continue
                    seen_row_ids.add(row.row_id)
                    results.append(_row_to_dict(row, score=2.0, match_type="exact_catalogue"))
            except Exception as e:
                print(f"[CanonicalStore] Exact lookup failed for {pattern}: {e}")

    # ── Strategy 2: Full-text search on nl_sentence (if no catalogue pattern)
    if not results and not catalogue_patterns:
        clean_query = re.sub(r'[|\-]{2,}', ' ', query).strip()
        sql = text("""
            SELECT row_id, table_id, section_title, page_start, page_end,
                   row_index, header_path, cells, numeric_cells, nl_sentence, note_refs,
                   ts_rank(to_tsvector('english', nl_sentence), plainto_tsquery('english', :query)) AS rank
            FROM canonical_table_rows
            WHERE tenant_id = :tenant_id
              AND nl_sentence IS NOT NULL
              AND plainto_tsquery('english', :query) @@ to_tsvector('english', nl_sentence)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        try:
            rows = db.execute(sql, {"tenant_id": tenant_id, "query": clean_query, "limit": limit}).fetchall()
            for row in rows:
                if row.row_id in seen_row_ids:
                    continue
                if numeric_filters and not _passes_numeric_filter(row.numeric_cells or {}, numeric_filters):
                    continue
                seen_row_ids.add(row.row_id)
                results.append(_row_to_dict(row, score=float(getattr(row, 'rank', 0.5)), match_type="fts"))
        except Exception as e:
            print(f"[CanonicalStore] FTS fallback failed: {e}")

    if results:
        print(f"[CanonicalStore] Found {len(results)} exact matches for patterns={catalogue_patterns}")
    return results


def _passes_numeric_filter(numeric_cells: dict, filters: Dict[str, Tuple[str, float]]) -> bool:
    """Check if a row's numeric cells pass all requested numeric filters."""
    for field_name, (op, value) in filters.items():
        cell_val = numeric_cells.get(field_name)
        if cell_val is None:
            # Try fuzzy field name match
            for k, v in numeric_cells.items():
                if field_name.lower() in k.lower():
                    cell_val = v
                    break
        if cell_val is None:
            continue  # Can't filter → allow through
        cell_val = float(cell_val)
        if op == ">=" and not (cell_val >= value): return False
        if op == "<=" and not (cell_val <= value): return False
        if op == ">"  and not (cell_val > value):  return False
        if op == "<"  and not (cell_val < value):  return False
        if op == "="  and not (abs(cell_val - value) < 0.01): return False
    return True


def _row_to_dict(row, score: float = 1.0, match_type: str = "exact") -> dict:
    """Convert a SQLAlchemy result row to a candidate dict compatible with retrieval.py."""
    cells = row.cells if isinstance(row.cells, dict) else (json.loads(row.cells) if row.cells else {})
    numeric = row.numeric_cells if isinstance(row.numeric_cells, dict) else (json.loads(row.numeric_cells) if row.numeric_cells else {})
    notes = row.note_refs if isinstance(row.note_refs, dict) else (json.loads(row.note_refs) if row.note_refs else {})
    headers = row.header_path if isinstance(row.header_path, dict) else (json.loads(row.header_path) if row.header_path else {})

    # Build a readable text representation for display/reranking
    text_parts = [f"{k}: {v}" for k, v in cells.items() if v]
    if row.section_title:
        text_parts.insert(0, f"[{row.section_title}]")
    text = "; ".join(text_parts)

    return {
        "id": row.row_id,
        "text": text,
        "score": score,
        "hybrid_score": score,
        "dense_score": 0.0,
        "file_type": "pdf",
        "table_group": row.table_id,
        "metadata": {
            "source": "",
            "section": row.row_index,
            "type": "table_row",
            "table_id": row.table_id,
            "section_title": row.section_title,
            "page_num": row.page_start,
            "cell_values": cells,
            "numeric_cells": numeric,
            "header_path": list(headers.values()) if headers else [],
            "row_index": row.row_index,
            "note_refs": notes,
            "match_type": match_type,
            "embedding_model": "exact_lookup",
        },
    }


# ---------------------------------------------------------------------------
# Aggregation query (0 tokens)
# ---------------------------------------------------------------------------

def aggregate_query(
    db: Session,
    tenant_id: str,
    numeric_filters: Dict[str, Tuple[str, float]] = None,
    section_title_hint: str = None,
    group_by_field: str = None,
    limit: int = 100,
) -> List[Dict]:
    """
    Return all rows matching numeric filters, optionally filtered by section.
    Used for "list all 200A models" → SQL scan of numeric_cells.
    """
    conditions = ["tenant_id = :tenant_id"]
    params: dict = {"tenant_id": tenant_id, "limit": limit}

    if section_title_hint:
        conditions.append("section_title ILIKE :section")
        params["section"] = f"%{section_title_hint}%"

    # Build numeric filter expressions using JSON path operators
    if numeric_filters:
        for i, (field_name, (op, value)) in enumerate(numeric_filters.items()):
            # Cast numeric_cells -> field -> float
            pg_op_map = {">=": ">=", "<=": "<=", ">": ">", "<": "<", "=": "="}
            pg_op = pg_op_map.get(op, "=")
            param_key = f"num_val_{i}"
            conditions.append(
                f"(numeric_cells->>'{field_name}')::float {pg_op} :{param_key}"
            )
            params[param_key] = value

    where_clause = " AND ".join(conditions)
    sql = text(f"""
        SELECT row_id, table_id, section_title, page_start, page_end,
               row_index, header_path, cells, numeric_cells, nl_sentence, note_refs
        FROM canonical_table_rows
        WHERE {where_clause}
        ORDER BY section_title, row_index
        LIMIT :limit
    """)

    results = []
    try:
        rows = db.execute(sql, params).fetchall()
        for row in rows:
            results.append(_row_to_dict(row, score=1.0, match_type="aggregation"))
        print(f"[CanonicalStore] Aggregation returned {len(results)} rows")
    except Exception as e:
        print(f"[CanonicalStore] Aggregation query failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Comparison query (fetch 2+ rows, LLM synthesizes diff)
# ---------------------------------------------------------------------------

def fetch_rows_for_comparison(
    db: Session,
    tenant_id: str,
    catalogue_patterns: List[str],
) -> List[Dict]:
    """
    Fetch exactly the rows needed for a comparison query.
    Returns one dict per catalogue pattern found.
    Used for "compare ECL2412SD and ECL3412SD".
    """
    results = []
    for pattern in catalogue_patterns:
        rows = exact_lookup(db, tenant_id, "", catalogue_patterns=[pattern], limit=1)
        if rows:
            results.append(rows[0])
    return results


# ---------------------------------------------------------------------------
# Builder: RichTable → CanonicalTableRow list
# ---------------------------------------------------------------------------

def rich_table_to_canonical_rows(
    rich_table,  # app.rag.table_engine.RichTable
    tenant_id: str,
    doc_id: str,
    doc_title: str = "",
) -> List[CanonicalTableRow]:
    """
    Convert a RichTable object (from table_engine.py) into a list of
    CanonicalTableRow objects ready for upsert into canonical_table_rows.
    """
    rows = []
    headers = rich_table.resolved_headers or rich_table.raw_headers
    page_start = rich_table.page_range[0] if rich_table.page_range else 1
    page_end   = rich_table.page_range[-1] if rich_table.page_range else page_start

    header_path = {f"col_{i}": h for i, h in enumerate(headers)}

    for r_idx, data_row in enumerate(rich_table.data_rows):
        # Build cells dict
        cells = {}
        for c_idx, cell in enumerate(data_row):
            h = headers[c_idx] if c_idx < len(headers) else f"col_{c_idx}"
            if cell.text.strip():
                cells[h] = cell.text.strip()

        if not cells:
            continue

        # Extract numeric values
        numeric_cells = _extract_numeric_cells(cells)

        # Resolve note references (circled numbers)
        note_refs = _extract_note_refs(cells)

        # Build NL sentence
        nl = rich_table.row_to_natural_language(data_row)
        if doc_title and doc_title not in nl:
            nl = f"[{doc_title}] {nl}"

        rows.append(CanonicalTableRow(
            tenant_id=tenant_id,
            doc_id=doc_id,
            table_id=rich_table.table_id,
            section_title=rich_table.section_title or "",
            page_start=page_start,
            page_end=page_end,
            row_index=r_idx,
            header_path=header_path,
            cells=cells,
            numeric_cells=numeric_cells,
            nl_sentence=nl,
            note_refs=note_refs,
        ))

    return rows


def _extract_numeric_cells(cells: Dict[str, str]) -> Dict[str, float]:
    """
    Parse numeric values from cell strings.
    "200A" → 200.0, "13-5/16" → 13.3125, "4-2/0 MCM" → 4.0 (gauge number)
    """
    numeric = {}
    num_pattern = re.compile(r'^(\d+(?:\.\d+)?)\s*([A-Za-z]*)')
    fraction_pattern = re.compile(r'^(\d+)-(\d+)/(\d+)$')  # e.g. 13-5/16

    for header, value in cells.items():
        if not value:
            continue
        val_str = value.strip()

        # Try fraction: "13-5/16"
        m = fraction_pattern.match(val_str)
        if m:
            whole, numer, denom = int(m.group(1)), int(m.group(2)), int(m.group(3))
            numeric[header] = round(whole + numer / denom, 4)
            continue

        # Try plain number or number+unit: "200", "200A", "34 in"
        m = num_pattern.match(val_str)
        if m and m.group(1):
            unit = m.group(2).lower()
            val = float(m.group(1))
            numeric[header] = val

            # Also store with normalized field name based on unit
            unit_field = {
                "a": "amps", "amp": "amps", "amps": "amps",
                "v": "volts", "kv": "kv",
                "w": "watts", "kw": "kw",
                "in": "inches", "inch": "inches",
                "mm": "mm",
            }.get(unit)
            if unit_field and unit_field not in numeric:
                numeric[unit_field] = val

    return numeric


def _extract_note_refs(cells: Dict[str, str]) -> Dict[str, str]:
    """Extract circled number references from cell values."""
    note_pattern = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩]')
    refs = {}
    for value in cells.values():
        for m in note_pattern.finditer(value or ""):
            refs[m.group()] = ""  # Placeholder; resolved at page level if footnotes available
    return refs


# ---------------------------------------------------------------------------
# Format helpers for LLM context
# ---------------------------------------------------------------------------

def format_rows_as_comparison_table(rows: List[Dict]) -> str:
    """
    Format two or more canonical row dicts as a side-by-side comparison HTML table.
    Used for COMPARISON query LLM context.
    """
    if not rows:
        return ""

    # Collect all unique headers
    all_headers = set()
    for row in rows:
        all_headers.update(row.get("metadata", {}).get("cell_values", {}).keys())
    all_headers = sorted(all_headers)

    # Build HTML table
    lines = ["<table border='1'>"]
    lines.append("<thead><tr><th>Field</th>" + "".join(
        f"<th>{row.get('metadata', {}).get('cell_values', {}).get('Catalogue Number', f'Item {i+1}')}</th>"
        for i, row in enumerate(rows)
    ) + "</tr></thead>")
    lines.append("<tbody>")
    for h in all_headers:
        cells_html = "".join(
            f"<td>{row.get('metadata', {}).get('cell_values', {}).get(h, '—')}</td>"
            for row in rows
        )
        lines.append(f"<tr><td><strong>{h}</strong></td>{cells_html}</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def format_rows_as_list(rows: List[Dict], section_title: str = "") -> str:
    """
    Format a list of canonical rows as a concise text list.
    Used for AGGREGATION query LLM context (or direct return without LLM).
    """
    if not rows:
        return "No matching records found."

    header = f"**{section_title}** — {len(rows)} matching models:\n" if section_title else f"{len(rows)} matching models:\n"
    lines = [header]
    for i, row in enumerate(rows, 1):
        cells = row.get("metadata", {}).get("cell_values", {})
        cat = cells.get("Catalogue Number", cells.get("catalogue number", f"Row {i}"))
        # Pick the most important fields (first 4)
        field_strs = [f"{k}={v}" for k, v in list(cells.items())[:4] if k != "Catalogue Number"]
        lines.append(f"  {i}. **{cat}** — {', '.join(field_strs)}")
    return "\n".join(lines)
