"""Rule-based query intent detection (sub-millisecond, deterministic, no LLM).

The intent only tunes retrieval depth and the answer template; retrieval
itself is the same for every intent, so a wrong guess degrades gracefully.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.identifiers import extract_identifiers


@dataclass(frozen=True)
class Intent:
    name: str
    top_k: int
    expand_section: bool
    guidance: str  # answer-structure hint for the LLM


INTENTS = {
    "alarm": Intent("alarm", 6, True,
                    "Explain what the alarm means, list documented causes, then the documented corrective steps in order. "
                    "Mention related alarms or probable root cause only if the sources state them."),
    "troubleshoot": Intent("troubleshoot", 8, True,
                           "Structure as: Symptom -> Documented causes -> Recommended checks/actions (in documented order)."),
    "procedure": Intent("procedure", 6, True,
                        "Reproduce the documented steps exactly, numbered, in order, including warnings and lock-out notes."),
    "parameter": Intent("parameter", 6, False,
                        "Give the exact value(s) with units, the tag/address they belong to, and the source row."),
    "history": Intent("history", 10, False,
                      "Summarise matching historical cases (date, equipment, root cause, action). Do not generalise beyond them."),
    "overview": Intent("overview", 10, True, "Give a structured overview using headings; cover every relevant source."),
    "comparison": Intent("comparison", 10, False, "Compare side by side; only use differences stated in the sources."),
    "general": Intent("general", 6, False, "Answer concisely."),
}

_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks?|thank you)\b[\s!.?]*$", re.I)
_RULES = [
    ("history", re.compile(r"\b(work ?orders?|WO-\d|history|historical|previous(ly)?|last time|in the past|how many times|recurr|happened before|past (cases|failures))\b", re.I)),
    ("comparison", re.compile(r"\b(difference|compare|comparison|versus|vs\.?)\b", re.I)),
    ("procedure", re.compile(r"\b(how (do|to|can) (i|we)?\s*(replace|install|calibrate|clean|reset|adjust|set ?up|configure|change|check)|procedure|steps?\b|preventive|PM-\d|maintenance (plan|schedule|task))", re.I)),
    ("troubleshoot", re.compile(r"\b(troubleshoot|not (working|starting|moving|responding)|fails?|failure|fault|problem|issue|defect|why (is|does|did)|what should i check|root cause|stopped|tripp?ed|leak|noise|vibration|sags?|runs?|orange peel|crater|blister|popping|mottl)\w*", re.I)),
    ("parameter", re.compile(r"\b(limit|setpoint|set point|range|value|tag|address|parameter|specification|spec|torque|pressure|temperature|rpm|kv|units?)\b", re.I)),
    ("overview", re.compile(r"\b(overview|summar(y|ise|ize)|explain everything|all (alarms|steps|parameters))\b", re.I)),
]
_ALARM_WORD = re.compile(r"\b(alarm|error code|fault code|alert)\b", re.I)


def is_greeting(query: str) -> bool:
    return bool(_GREETING.match(query or ""))


_ALARM_VERBS = re.compile(r"\b(mean|meaning|means|fix|reset|clear|acknowledge|cause[sd]?|active|raised|triggered)\b", re.I)


def detect_intent(query: str) -> Intent:
    q = query or ""
    has_id = bool(extract_identifiers(q))
    for name, rx in _RULES[:2]:  # history / comparison
        if rx.search(q):
            return INTENTS[name]
    if has_id and (_ALARM_WORD.search(q) or _ALARM_VERBS.search(q)):
        return INTENTS["alarm"]
    if _RULES[2][1].search(q):  # procedure
        return INTENTS["procedure"]
    if _ALARM_WORD.search(q):
        return INTENTS["alarm"]
    for name, rx in _RULES[3:]:
        if rx.search(q):
            return INTENTS[name]
    return INTENTS["alarm"] if has_id else INTENTS["general"]
