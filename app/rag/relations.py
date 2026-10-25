"""Lightweight relationship graph over identifiers (alarms, tags, equipment, WOs, parts).

Every chunk stores the identifiers it mentions (GIN-indexed array). Two
identifiers are related when they are documented together in the same passage
(alarm manual "related alarms", troubleshooting rows, work orders, tag lists).
This gives alarm-relationship / root-cause context and "where else is this
documented" links without running a separate graph database.
"""

from __future__ import annotations

import re
from typing import Dict, Sequence

from sqlalchemy import text as sql

from app.db import SessionLocal
from app.rag.chunking import _doc_ids

_RELATED = sql("""
    WITH hit AS (
        SELECT identifiers, content_type FROM chunks
        WHERE tenant_id = :t AND identifiers && :ids
        LIMIT :scan
    )
    SELECT i AS identifier, count(*) AS mentions,
           count(*) FILTER (WHERE content_type = 'record') AS records
    FROM hit, unnest(identifiers) AS i
    WHERE NOT (i = ANY(:ids))
    GROUP BY i
    ORDER BY mentions DESC, i
    LIMIT :lim
""")

_COUNTS = sql("""
    SELECT count(*) AS total, count(*) FILTER (WHERE content_type = 'record') AS records,
           count(DISTINCT document_id) AS documents
    FROM (SELECT content_type, document_id FROM chunks WHERE tenant_id = :t AND identifiers && :ids LIMIT :scan) s
""")

# Where else the asked-about codes are documented (one passage per document section).
_LOCATIONS = sql("""
    SELECT DISTINCT ON (c.document_id, coalesce(c.section_title, ''))
           c.id, c.document_id, d.name, c.section_title, c.page, c.content_type, left(c.text, 220) AS snippet
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.tenant_id = :t AND c.identifiers && :ids AND NOT (c.id = ANY(:exclude))
    ORDER BY c.document_id, coalesce(c.section_title, ''), c.ord
    LIMIT :lim
""")

_DOC_NAMES = sql("""
    SELECT DISTINCT d.name FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.tenant_id = :t AND c.identifiers && :ids LIMIT 200
""")

# A short human label for a code: the heading that names it, or the text after it in a table row.
_LABELS = sql("""
    SELECT DISTINCT ON (i) i AS identifier, c.section_title, c.text
    FROM chunks c, unnest(:cand) AS i
    WHERE c.tenant_id = :t AND c.identifiers @> ARRAY[i]
    ORDER BY i, (coalesce(c.section_title, '') ILIKE '%' || i || '%') DESC, (c.content_type = 'table_row') DESC, c.id
""")


def _label(identifier: str, section: str, text: str) -> str:
    """Only labels the documents state: the heading that names the code, or the
    Message/Description field of the table row whose Code/Alarm/Tag cell is the code."""
    if section and identifier.upper() in section.upper():
        tail = section.upper().split(identifier.upper(), 1)[1]
        label = section[len(section) - len(tail):].strip(" -:–")
        if label:
            return label[:80]
    m = re.search(r"(?:^|; )(?:code|alarm|tag|parameter|part)[^:;]{0,12}: " + re.escape(identifier)
                  + r"\b.*?(?:message|description|symptom|name): ([^;\n]{3,80})", text or "", re.I)
    return m.group(1).strip(" .;") if m else ""


_PRIORITY = ["code", "tag", "address", "part", "work_order", "robot"]


def _kind(identifier: str) -> str:
    i = identifier.upper()
    if re.fullmatch(r"R\d{1,3}", i):
        return "robot"
    if re.fullmatch(r"WO-.*|[A-Z]{2,4}-\d{2,4}-\d+", i):
        return "work_order"
    if re.fullmatch(r"\d[\d-]+", i):
        return "part"
    if i.startswith(("DB", "%")) and re.search(r"\.DB|%", i):
        return "address"
    if "_" in i or "." in i:
        return "tag"
    return "code"


def related_identifiers(tenant_id: str, ids: Sequence[str], limit: int = 10, scan: int = 5000,
                        exclude_chunks: Sequence[int] = ()) -> dict:
    ids = [i for i in ids if i]
    empty = {"identifiers": list(ids), "related": [], "locations": [], "mentions": 0, "records": 0, "documents": 0}
    if not ids:
        return empty
    with SessionLocal() as db:
        # Codes that come from file names (machine model "P-500" etc.) tag every passage of a
        # manual, so they are not relationships: leave them out on both sides.
        doc_level = {i for (name,) in db.execute(_DOC_NAMES, {"t": tenant_id, "ids": list(ids)}).all()
                     for i in _doc_ids(name)}
        ids = [i for i in ids if i not in doc_level]
        if not ids:
            return empty
        params = {"t": tenant_id, "ids": list(ids), "scan": scan}
        rows = db.execute(_RELATED, {**params, "lim": limit * 3}).all()
        counts = db.execute(_COUNTS, params).one()
        # The most specific code in the question decides what "related" means (an alarm ->
        # related alarms). Robot numbers / work orders are shown only when they are what was asked.
        primary = min((_kind(i) for i in ids), key=_PRIORITY.index)
        # For a robot or a work order, the useful relation is the alarms recorded with it.
        wanted = "code" if primary in {"robot", "work_order"} else primary
        pool = [r for r in rows if r.identifier not in doc_level and _kind(r.identifier) not in {"robot", "work_order"}]
        pool.sort(key=lambda r: (_kind(r.identifier) != wanted, -r.mentions, r.identifier))
        candidates = pool[:limit]
        labels: Dict[str, str] = {}
        if candidates:
            for r in db.execute(_LABELS, {"t": tenant_id, "cand": [c.identifier for c in candidates]}).all():
                labels[r.identifier] = _label(r.identifier, r.section_title or "", r.text)
        locations = db.execute(_LOCATIONS, {**params, "exclude": list(exclude_chunks) or [-1], "lim": 6}).all()
    return {
        "identifiers": list(ids),
        "related": [{"identifier": r.identifier, "mentions": r.mentions, "records": r.records,
                     "label": labels.get(r.identifier, "")} for r in candidates],
        "locations": [{"chunk_id": r.id, "document_id": r.document_id, "document": r.name, "section": r.section_title,
                       "page": r.page, "content_type": r.content_type, "snippet": r.snippet} for r in locations],
        "mentions": counts.total, "records": counts.records, "documents": counts.documents,
    }
