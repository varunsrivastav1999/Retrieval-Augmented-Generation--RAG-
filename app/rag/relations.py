"""Lightweight relationship graph over identifiers (alarms, tags, equipment, WOs).

Every chunk stores the identifiers it mentions (GIN-indexed array). Two
identifiers are related when they co-occur in documented text (alarm manual
"related alarms", work orders, tag lists). This gives alarm-relationship and
root-cause context without running a separate graph database.
"""

from __future__ import annotations

from typing import List, Sequence

from sqlalchemy import text as sql

from app.db import SessionLocal

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


def related_identifiers(tenant_id: str, ids: Sequence[str], limit: int = 12, scan: int = 5000) -> dict:
    ids = [i for i in ids if i]
    if not ids:
        return {"identifiers": [], "related": [], "mentions": 0, "records": 0, "documents": 0}
    with SessionLocal() as db:
        rows = db.execute(_RELATED, {"t": tenant_id, "ids": list(ids), "lim": limit, "scan": scan}).all()
        counts = db.execute(_COUNTS, {"t": tenant_id, "ids": list(ids), "scan": scan}).one()
    related: List[dict] = [{"identifier": r.identifier, "mentions": r.mentions, "records": r.records} for r in rows]
    return {"identifiers": list(ids), "related": related, "mentions": counts.total,
            "records": counts.records, "documents": counts.documents}
