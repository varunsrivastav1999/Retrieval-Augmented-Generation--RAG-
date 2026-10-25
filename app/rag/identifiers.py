"""Extraction of industrial identifiers (alarm codes, tags, PLC addresses, work orders).

Maintenance questions usually hinge on an exact identifier ("SRVO-050",
"OVN2_Z3_TEMP_PV", "DB120.DBD24", "WO-2024-0013245"). Dense embeddings blur
these strings, so they are extracted at ingestion time (stored per chunk) and
at query time (matched exactly). Pure functions, no I/O.
"""

from __future__ import annotations

import re
from typing import List

# Order matters: more specific patterns first. Each pattern must contain at
# least one digit so ordinary words never match.
_PATTERNS = [
    # Siemens-style PLC addresses: DB120.DBD24, DB5.DBX2.0, %MW100, I0.1, Q4.7
    r"\bDB\d+\.DB[XBWD]\d+(?:\.\d+)?\b",
    r"%[IQM][XBWD]?\d+(?:\.\d+)?\b",
    # Allen-Bradley style: N7:0, B3:1/2, Local:1:I.Data.0
    r"\b[NBFTCRS]\d+:\d+(?:/\d+)?\b",
    # Work orders / tickets: WO-2024-0013245, WO123456, NCR-22-001
    r"\b[A-Z]{2,4}-\d{2,4}-\d{2,8}\b",
    # Tags with underscores/dots: OVN2_Z3_TEMP_PV, BOOTH1.EXH_FAN.MTR_TEMP
    r"\b[A-Za-z][A-Za-z0-9]*(?:[_.][A-Za-z0-9]+){1,6}\b",
    # Alarm / error codes: SRVO-050, PNT1-128, PLC-2041, E-102, F0021, AL123
    r"\b[A-Z][A-Z0-9]{0,9}(?:-\d{1,6}){1,2}[A-Z]?\b",
    r"\b[A-Z]{1,4}\d{2,6}[A-Z0-9-]*\b",
    # Equipment numbers like R07, PLC-01 handled above; bare robot ids:
    r"\bR\d{1,3}\b",
    # Numeric part numbers: 8301-5662-59, 0100-2233-445-1 (dates are rejected below)
    r"\b\d{3,6}(?:-\d{2,6}){2,3}\b",
]
_ID_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))
_HAS_DIGIT = re.compile(r"\d")
_PART_NO = re.compile(r"\d{3,6}(?:-\d{2,6}){2,3}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Tokens that look like identifiers but are just numbers with units or versions.
_STOP = {"V1", "V2", "V3"}


def extract_identifiers(text: str, limit: int = 64) -> List[str]:
    """Return unique identifiers in order of appearance (case preserved)."""
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for match in _ID_RE.finditer(text):
        token = match.group(0).strip(".:/")
        if len(token) < 3 or not _HAS_DIGIT.search(token) or _DATE.fullmatch(token):
            continue
        # Pure numbers / decimals / dates are values, not identifiers; numeric part numbers are kept.
        if re.fullmatch(r"[\d.,:/-]+", token) and not _PART_NO.fullmatch(token):
            continue
        key = token.upper()
        if key in seen or key in _STOP:
            continue
        seen.add(key)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def normalize_identifier(token: str) -> str:
    """Canonical form used for exact matching (case-insensitive)."""
    return token.strip().upper()
