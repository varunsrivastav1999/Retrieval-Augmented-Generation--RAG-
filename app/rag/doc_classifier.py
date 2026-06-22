"""
=============================================================================
 Enterprise Level RAG: Document Format Classifier & Content Normalizer v1.0
=============================================================================
 Detects and normalizes ALL document content formats:
   - MCQ (Multiple Choice Questions)
   - Fill-in-the-blank
   - True/False
   - Match the column
   - Short answer / essay prompts
   - Forms / label-value pairs
   - Technical tables (covered by table_engine.py)
   - Paragraphs / prose
   - Ordered / unordered lists
   - Code blocks
   - Mathematical equations
   - Definitions / glossaries
   - Legal / regulatory clauses

 Each format is extracted as a structured dict with:
   - content_type: the format label
   - nl_sentence: embeddable natural-language representation
   - structured_data: JSON-serializable metadata
   - chunk_text: text to be stored in document_chunks
   - search_tags: list of strings for BM25 boosting
=============================================================================
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Content type constants
# ---------------------------------------------------------------------------

class ContentType:
    PARAGRAPH        = "paragraph"
    MCQ              = "mcq"
    FILL_BLANK       = "fill_blank"
    TRUE_FALSE       = "true_false"
    MATCH_COLUMN     = "match_column"
    SHORT_ANSWER     = "short_answer"
    ESSAY_PROMPT     = "essay_prompt"
    FORM_FIELD       = "form_field"
    TABLE_ROW        = "table_row"
    LIST_ITEM        = "list_item"
    CODE_BLOCK       = "code_block"
    EQUATION         = "equation"
    DEFINITION       = "definition"
    LEGAL_CLAUSE     = "legal_clause"
    HEADING          = "heading"
    CAPTION          = "caption"
    FOOTNOTE         = "footnote"
    KEY_VALUE        = "key_value"           # "Field Name: value" pairs
    SPECIFICATION    = "specification"       # Engineering specs


@dataclass
class DetectedContent:
    """A single detected content block from a document page."""
    content_type: str
    raw_text: str
    nl_sentence: str
    structured_data: dict = field(default_factory=dict)
    chunk_text: str = ""
    search_tags: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self):
        if not self.chunk_text:
            self.chunk_text = self.raw_text


# ---------------------------------------------------------------------------
# MCQ Detection & Parsing
# ---------------------------------------------------------------------------

# Patterns that indicate MCQ checkmarks (correct answer markers)
_MCQ_TICK_RE = re.compile(r'[✓✔☑☒◉]|\[x\]|\[X\]|\(x\)|\(X\)')

# MCQ option prefixes: (A) (a) A. 1. (1) — up to 8 options
_MCQ_OPTION_RE = re.compile(
    r'^(?:\(?([A-Ha-h1-8])[.)]\s*|([A-Ha-h1-8])\.\s*)(.+?)(?:\s*[✓✔☑☒◉]|\s*\[x\]|\s*\[X\])?$',
    re.MULTILINE
)

# Question stem: ends with "?" or a colon, followed by options on next line
_MCQ_QUESTION_RE = re.compile(
    r'^(Q(?:uestion)?\.?\s*\d+\.?\s*|(?:\d+[.)])\s*)(.+?[?:])\s*$',
    re.MULTILINE | re.IGNORECASE
)


def detect_mcq(text: str) -> Optional[DetectedContent]:
    """
    Detect MCQ blocks and extract question, options, and correct answer.
    Returns DetectedContent if MCQ is detected, else None.
    """
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return None

    # Count option lines
    option_lines = [l for l in lines if _MCQ_OPTION_RE.match(l.strip())]
    if len(option_lines) < 2:
        return None

    # Find question (line before options or explicit Q# pattern)
    question = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _MCQ_QUESTION_RE.match(stripped)
        if m:
            question = m.group(2).strip()
            break
        if stripped and not _MCQ_OPTION_RE.match(stripped) and i < len(lines) - 1:
            if _MCQ_OPTION_RE.match(lines[min(i + 1, len(lines) - 1)].strip()):
                question = stripped
                break

    if not question:
        question = lines[0].strip()

    # Parse all options
    options: Dict[str, str] = {}
    correct_option: Optional[str] = None
    correct_text: Optional[str] = None

    for line in lines:
        m = _MCQ_OPTION_RE.match(line.strip())
        if m:
            label = (m.group(1) or m.group(2)).upper()
            option_text = m.group(3).strip()
            options[label] = option_text
            if _MCQ_TICK_RE.search(line):
                correct_option = label
                correct_text = option_text

    if len(options) < 2:
        return None

    structured = {
        "question": question,
        "options": options,
        "correct_option": correct_option,
        "correct_answer": correct_text,
        "num_options": len(options),
    }

    # Build NL sentence for embedding
    if correct_text:
        nl = f"The correct answer to '{question}' is {correct_text} (Option {correct_option})."
    else:
        option_str = "; ".join(f"({k}) {v}" for k, v in options.items())
        nl = f"Question: {question} Options: {option_str}."

    # Build chunk text (structured for retrieval)
    chunk_lines = [f"[MCQ] {question}"]
    for label, opt_text in options.items():
        marker = " ✓" if label == correct_option else ""
        chunk_lines.append(f"  ({label}) {opt_text}{marker}")
    if correct_text:
        chunk_lines.append(f"[CORRECT ANSWER] ({correct_option}) {correct_text}")

    return DetectedContent(
        content_type=ContentType.MCQ,
        raw_text=text,
        nl_sentence=nl,
        structured_data=structured,
        chunk_text="\n".join(chunk_lines),
        search_tags=[question[:50], correct_text or "", "mcq", "multiple choice"],
        confidence=0.95 if correct_option else 0.75,
    )


# ---------------------------------------------------------------------------
# Fill-in-the-Blank Detection & Parsing
# ---------------------------------------------------------------------------

# Blank patterns: _____, ........, [blank], _blank_, (    ), [____]
_BLANK_RE = re.compile(r'_{3,}|\.{4,}|\[_+\]|\[blank\]|\(_+\)|\[answer\]', re.IGNORECASE)

# Answer on same line after colon or in brackets: "Answer: Ohm" or "(Ans: V=IR)"
_ANSWER_INLINE_RE = re.compile(
    r'\b(?:answer|ans|solution|sol)[\s.:]+([^\n]{1,100})',
    re.IGNORECASE
)


def detect_fill_blank(text: str) -> Optional[DetectedContent]:
    """
    Detect fill-in-the-blank questions and extract template + answer.
    """
    blanks = _BLANK_RE.findall(text)
    if not blanks:
        return None

    # Try to find the answer
    answer = None
    answer_match = _ANSWER_INLINE_RE.search(text)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Build the question template (blanks → _____)
    template = _BLANK_RE.sub('_____', text)
    # Remove the answer part from template
    if answer_match:
        template = template[:answer_match.start()].strip()

    blank_count = len(blanks)

    structured = {
        "question_template": template.strip(),
        "blank_count": blank_count,
        "answer": answer,
    }

    if answer:
        # Build filled-in sentence
        filled = _BLANK_RE.sub(answer, template, count=1)
        nl = filled.strip()
        if not nl.endswith('.'):
            nl += '.'
    else:
        nl = f"Fill in the blank: {template.strip()}"

    chunk_lines = [
        f"[FILL_BLANK] {template.strip()}",
    ]
    if answer:
        chunk_lines.append(f"[ANSWER] {answer}")

    return DetectedContent(
        content_type=ContentType.FILL_BLANK,
        raw_text=text,
        nl_sentence=nl,
        structured_data=structured,
        chunk_text="\n".join(chunk_lines),
        search_tags=["fill in the blank", "answer", template[:50]],
        confidence=0.9 if answer else 0.7,
    )


# ---------------------------------------------------------------------------
# True/False Detection
# ---------------------------------------------------------------------------

_TF_RE = re.compile(
    r'^\s*(?:(?:Q(?:uestion)?\.?\s*\d+\.?\s*)?)(.+?)\s*(?:\n|\s{3,})'
    r'(?:True|False|T|F)\s*[✓✔☑✗☒✘×]?\s*$',
    re.MULTILINE | re.IGNORECASE
)
_TF_ANSWER_RE = re.compile(r'\b(True|False|T\b|F\b)\b', re.IGNORECASE)
_TF_TICK_RE   = re.compile(r'[✓✔☑]')
_TF_CROSS_RE  = re.compile(r'[✗☒✘×]')


def detect_true_false(text: str) -> Optional[DetectedContent]:
    """Detect True/False statements."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 1:
        return None

    # Look for "True" or "False" keyword with optional tick/cross
    tf_line = None
    for line in lines:
        if _TF_ANSWER_RE.search(line):
            tf_line = line
            break

    if tf_line is None:
        return None

    # Determine answered value
    correct = None
    if _TF_TICK_RE.search(tf_line):
        m = _TF_ANSWER_RE.search(tf_line)
        correct = m.group(1).lower() if m else None
    elif _TF_CROSS_RE.search(tf_line):
        m = _TF_ANSWER_RE.search(tf_line)
        if m:
            stated = m.group(1).lower()
            correct = "false" if stated == "true" else "true"

    statement = lines[0] if len(lines) > 1 else text.strip()

    structured = {
        "statement": statement,
        "correct_value": correct,
    }

    nl = f"The statement '{statement}' is {correct}." if correct else f"True/False: {statement}"

    return DetectedContent(
        content_type=ContentType.TRUE_FALSE,
        raw_text=text,
        nl_sentence=nl,
        structured_data=structured,
        chunk_text=f"[TRUE_FALSE] {statement}\n[ANSWER] {correct or 'unknown'}",
        search_tags=["true false", statement[:50]],
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# Form Field (Label: Value) Detection
# ---------------------------------------------------------------------------

# Patterns: "Field Name: value", "Name .............. value", "Name [  value  ]"
_FORM_LABEL_VALUE_RE = re.compile(
    r'^([A-Za-z][A-Za-z0-9\s/\-]{2,40}?)\s*[:\|]\s*(.{1,200}?)(?:\s*$|\s{3,})',
    re.MULTILINE
)

# Form continuation patterns (dotted leaders)
_DOTTED_LEADER_RE = re.compile(
    r'^([A-Za-z][A-Za-z0-9\s/\-]{2,40}?)\s*[.·]{3,}\s*(.{1,200}?)$',
    re.MULTILINE
)


def detect_form_fields(text: str) -> Optional[DetectedContent]:
    """
    Detect form-style label:value pairs.
    Returns DetectedContent with structured_data = {"fields": {"label": "value"}}.
    """
    fields: Dict[str, str] = {}

    for pattern in [_FORM_LABEL_VALUE_RE, _DOTTED_LEADER_RE]:
        for m in pattern.finditer(text):
            label = m.group(1).strip().rstrip('.:')
            value = m.group(2).strip()
            if len(label) < 2 or len(value) < 1:
                continue
            # Skip if looks like a table row
            if '|' in label or value.count('|') > 1:
                continue
            fields[label] = value

    if len(fields) < 2:
        return None

    structured = {"fields": fields}

    # NL sentence: "Name=value; Date=value; ..."
    nl = "; ".join(f"{k}: {v}" for k, v in fields.items()) + "."

    chunk_lines = ["[FORM]"]
    for k, v in fields.items():
        chunk_lines.append(f"  {k}: {v}")

    return DetectedContent(
        content_type=ContentType.FORM_FIELD,
        raw_text=text,
        nl_sentence=nl,
        structured_data=structured,
        chunk_text="\n".join(chunk_lines),
        search_tags=list(fields.keys())[:5],
        confidence=0.80,
    )


# ---------------------------------------------------------------------------
# Definition / Glossary Detection
# ---------------------------------------------------------------------------

_DEFINITION_RE = re.compile(
    r'^([A-Z][A-Za-z\s\-/]{1,60}?)(?:\s*[:\-—]\s*|\s{2,})([A-Z].{10,300}?)(?:\.|$)',
    re.MULTILINE
)


def detect_definition(text: str) -> Optional[DetectedContent]:
    """Detect glossary/definition entries."""
    lines = text.strip().split('\n')
    if len(lines) > 10:
        return None  # Too long for a definition

    for line in lines:
        m = _DEFINITION_RE.match(line.strip())
        if m:
            term = m.group(1).strip().rstrip(':-—')
            definition = m.group(2).strip()
            structured = {"term": term, "definition": definition}
            nl = f"{term} means {definition}"
            return DetectedContent(
                content_type=ContentType.DEFINITION,
                raw_text=text,
                nl_sentence=nl,
                structured_data=structured,
                chunk_text=f"[DEFINITION] {term}: {definition}",
                search_tags=[term, "definition", "glossary"],
                confidence=0.80,
            )
    return None


# ---------------------------------------------------------------------------
# Specification line detection (engineering specs)
# ---------------------------------------------------------------------------

_SPEC_RE = re.compile(
    r'^([A-Za-z][A-Za-z0-9\s/\-\(\)]{2,50}?)\s*[:\-]\s*(\d[\d.,/\s]*(?:A|V|W|kW|kV|Hz|rpm|in|mm|kg|lb|%|°[CF])[^\n]*)',
    re.MULTILINE
)


def detect_specification(text: str) -> Optional[DetectedContent]:
    """Detect technical specification lines with values and units."""
    specs: Dict[str, str] = {}
    for m in _SPEC_RE.finditer(text):
        label = m.group(1).strip().rstrip(':-')
        value = m.group(2).strip()
        specs[label] = value

    if len(specs) < 2:
        return None

    structured = {"specs": specs}
    nl = "Technical specifications: " + "; ".join(f"{k}={v}" for k, v in specs.items()) + "."
    chunk_lines = ["[SPECIFICATION]"] + [f"  {k}: {v}" for k, v in specs.items()]

    return DetectedContent(
        content_type=ContentType.SPECIFICATION,
        raw_text=text,
        nl_sentence=nl,
        structured_data=structured,
        chunk_text="\n".join(chunk_lines),
        search_tags=list(specs.keys())[:5] + ["specification", "technical"],
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# Equation detection
# ---------------------------------------------------------------------------

_EQUATION_RE = re.compile(
    r'(?:[A-Za-z]\s*=\s*[A-Za-z0-9\s\+\-\*/\^\(\)\[\]\{\}\.\,]+|'
    r'\b(?:Formula|Equation|Where|Given)\b.{1,200})',
    re.IGNORECASE
)


def detect_equation(text: str) -> Optional[DetectedContent]:
    """Detect mathematical/engineering equations."""
    m = _EQUATION_RE.search(text)
    if not m:
        return None
    if len(text.strip()) > 500:  # Too long to be a standalone equation
        return None

    eq_text = m.group(0).strip()
    nl = f"Equation: {eq_text}"

    return DetectedContent(
        content_type=ContentType.EQUATION,
        raw_text=text,
        nl_sentence=nl,
        structured_data={"equation": eq_text},
        chunk_text=f"[EQUATION] {eq_text}",
        search_tags=["equation", "formula", "calculation"],
        confidence=0.75,
    )


# ---------------------------------------------------------------------------
# Master classifier
# ---------------------------------------------------------------------------

def classify_and_enrich_text_block(
    text: str,
    page_num: int = 1,
    section_title: str = "",
    is_table: bool = False,
) -> DetectedContent:
    """
    Master classifier. Takes a raw text block and returns the most appropriate
    DetectedContent with enriched metadata.

    Call this from ingestion.py instead of just storing raw text.
    """
    if is_table:
        return DetectedContent(
            content_type=ContentType.TABLE_ROW,
            raw_text=text,
            nl_sentence=text,
            chunk_text=text,
            search_tags=["table"],
        )

    stripped = text.strip()
    if not stripped:
        return DetectedContent(
            content_type=ContentType.PARAGRAPH,
            raw_text=text,
            nl_sentence=text,
            chunk_text=text,
        )

    # Priority order: most specific → most general
    detectors = [
        detect_mcq,
        detect_fill_blank,
        detect_true_false,
        detect_specification,
        detect_form_fields,
        detect_definition,
        detect_equation,
    ]

    for detector in detectors:
        try:
            result = detector(stripped)
            if result and result.confidence >= 0.7:
                # Prepend section title if available
                if section_title and section_title not in result.nl_sentence:
                    result.nl_sentence = f"[{section_title}] {result.nl_sentence}"
                    result.search_tags.append(section_title)
                return result
        except Exception as e:
            print(f"[DocClassifier] Detector {detector.__name__} failed: {e}")

    # Default: plain paragraph
    return DetectedContent(
        content_type=ContentType.PARAGRAPH,
        raw_text=text,
        nl_sentence=stripped if len(stripped) < 300 else stripped[:300] + "...",
        chunk_text=stripped,
        search_tags=[section_title] if section_title else [],
    )


def enrich_page_text(
    raw_text: str,
    page_num: int = 1,
    section_title: str = "",
) -> List[DetectedContent]:
    """
    Split a page's text into content blocks and classify each one.
    Returns a list of DetectedContent objects, one per logical block.

    Used by parsers.py to produce richer, typed content.
    """
    # Split on double newline (paragraph breaks)
    blocks = re.split(r'\n{2,}', raw_text)
    results = []

    for block in blocks:
        block = block.strip()
        if len(block) < 5:
            continue
        detected = classify_and_enrich_text_block(
            block,
            page_num=page_num,
            section_title=section_title,
        )
        results.append(detected)

    return results


# ---------------------------------------------------------------------------
# chunk_text generator for ingestion.py
# ---------------------------------------------------------------------------

def detected_content_to_chunk(
    content: DetectedContent,
    doc_title: str = "",
    page_num: int = 1,
    section_title: str = "",
) -> dict:
    """
    Convert a DetectedContent into the chunk dict format expected by ingestion.py.
    Maps directly to the pending_chunks list structure.
    """
    return {
        "text": content.chunk_text,
        "nl_text": content.nl_sentence,
        "content_type": content.content_type,
        "is_parent": True,
        "parent_idx": None,
        "child_idx": None,
        "table_group": None,
        "table_id": None,
        "section_title": section_title or content.structured_data.get("section", ""),
        "cell_values": None,
        "header_path": [],
        "row_index": None,
        "search_tags": content.search_tags,
        "structured_data": content.structured_data,
        "doc_metadata_extra": {
            "content_type": content.content_type,
            "confidence": content.confidence,
            "structured_data": content.structured_data,
            "search_tags": content.search_tags,
        },
    }
