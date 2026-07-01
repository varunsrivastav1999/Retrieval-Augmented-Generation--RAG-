"""
=============================================================================
 Enterprise Level RAG v6.0: Hierarchical Document Structure Builder
=============================================================================
 Parses document pages into a tree structure:
   Document → Chapter → Section → Subsection → Paragraph → Chunk

 Key capabilities:
   - Heading detection via regex + font-size heuristics + Docling markers
   - TOC-based structure extraction
   - Cross-page section merging
   - heading_path breadcrumbs for every node
   - Section embedding generation for section-level search

 All operations are synchronous, CPU-only, and 100% offline.
=============================================================================
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Heading:
    """A detected heading in a document."""
    text: str
    level: int  # 1=chapter, 2=section, 3=subsection, 4=sub-subsection
    page_num: int
    char_offset: int = 0  # Offset within page text


@dataclass
class DocumentNode:
    """A node in the hierarchical document tree."""
    node_id: str
    level: str  # "document" | "chapter" | "section" | "subsection" | "paragraph"
    title: str
    content: str  # Full text content of this node (excluding children)
    page_start: int
    page_end: int
    parent_id: Optional[str]
    children: List["DocumentNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    heading_path: List[str] = field(default_factory=list)
    chunk_count: int = 0  # Populated after chunking


# ---------------------------------------------------------------------------
# Heading Detection Patterns
# ---------------------------------------------------------------------------

# Numbered headings: "1.", "1.1", "1.1.1", "Chapter 1", "Section 2.3"
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:chapter|ch|section|sec)[\s.:]+(\d+(?:\.\d+)*)"  # "Chapter 1", "Section 2.3"
    r"|(\d+(?:\.\d+)*)\s*[.:)]\s+\S"                     # "1. Title", "2.3: Title"
    r"|(\d+(?:\.\d+)+)\s+\S"                               # "1.1 Title" (must have sub-number)
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# ALL-CAPS headings (common in manuals): "USB COMMUNICATION", "TROUBLESHOOTING"
_CAPS_HEADING_RE = re.compile(
    r"^([A-Z][A-Z\s\-&/]{4,60})$",
    re.MULTILINE,
)

# Markdown-style headings: "# Title", "## Subtitle"
_MARKDOWN_HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+)$",
    re.MULTILINE,
)

# Docling heading markers (if present)
_DOCLING_HEADING_RE = re.compile(
    r"^(?:\[HEADING(?:\s+LEVEL\s*=?\s*(\d+))?\])\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)

# Bold/underline heading heuristics
_BOLD_HEADING_RE = re.compile(
    r"^\*\*(.+?)\*\*$",
    re.MULTILINE,
)

# Common manual section titles
_KNOWN_SECTION_TITLES = frozenset([
    "introduction", "overview", "installation", "configuration", "setup",
    "wiring", "connection", "communication", "programming", "operation",
    "maintenance", "troubleshooting", "specifications", "parameters",
    "commands", "registers", "alarms", "errors", "error codes",
    "safety", "warnings", "precautions", "appendix", "glossary",
    "dimensions", "accessories", "parts", "ordering", "warranty",
    "features", "description", "theory", "principle", "procedure",
    "calibration", "testing", "diagnostics", "faq",
    "usb", "ethernet", "serial", "modbus", "profinet", "profibus",
    "inputs", "outputs", "analog", "digital", "power supply",
])


def _heading_level_from_number(number_str: str) -> int:
    """Determine heading level from a numbered section string like '1', '1.2', '1.2.3'."""
    parts = number_str.split(".")
    depth = len(parts)
    if depth == 1:
        return 1  # Chapter level
    elif depth == 2:
        return 2  # Section level
    elif depth == 3:
        return 3  # Subsection level
    else:
        return min(depth, 4)  # Sub-subsection


def _is_likely_heading(line: str) -> bool:
    """Quick heuristic: is this line likely a heading (short, no sentence-ending punctuation)?"""
    stripped = line.strip()
    if not stripped or len(stripped) < 2:
        return False
    if len(stripped) > 120:
        return False
    # Headings typically don't end with periods (unless abbreviations)
    if stripped.endswith(".") and not re.search(r"\d\.$", stripped):
        # Allow numbered headings like "1.2."
        if not re.match(r"^\d+(\.\d+)*\.\s", stripped):
            return False
    return True


# ---------------------------------------------------------------------------
# Heading Extractor
# ---------------------------------------------------------------------------

def extract_headings(text: str, page_num: int = 0) -> List[Heading]:
    """
    Extract headings from a page of text using multiple detection strategies.
    Returns headings sorted by their character offset within the text.
    """
    headings: List[Heading] = []
    seen_offsets: set = set()

    def _add(text_val: str, level: int, offset: int):
        text_val = text_val.strip()
        if not text_val or len(text_val) < 2 or offset in seen_offsets:
            return
        seen_offsets.add(offset)
        headings.append(Heading(text=text_val, level=level, page_num=page_num, char_offset=offset))

    # Strategy 1: Docling heading markers (highest confidence)
    for m in _DOCLING_HEADING_RE.finditer(text):
        level = int(m.group(1)) if m.group(1) else 2
        _add(m.group(2), level, m.start())

    # Strategy 2: Markdown headings
    for m in _MARKDOWN_HEADING_RE.finditer(text):
        level = len(m.group(1))  # Number of # characters
        _add(m.group(2), level, m.start())

    # Strategy 3: Numbered headings (1., 1.1, Chapter 1, Section 2.3)
    for m in _NUMBERED_HEADING_RE.finditer(text):
        # Get the full line containing this match
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].strip()

        if not _is_likely_heading(line):
            continue

        # Determine level from number
        num_str = m.group(1) or m.group(2) or m.group(3) or ""
        level = _heading_level_from_number(num_str) if num_str else 2
        _add(line, level, m.start())

    # Strategy 4: ALL-CAPS lines (manual-style headings)
    for m in _CAPS_HEADING_RE.finditer(text):
        candidate = m.group(1).strip()
        # Filter out false positives (model numbers, abbreviations, table headers)
        if len(candidate) < 5 or len(candidate.split()) < 2:
            # Single word ALL CAPS is not a heading unless it's a known section title
            if candidate.lower() not in _KNOWN_SECTION_TITLES:
                continue
        # ALL-CAPS headings are typically chapter or section level
        _add(candidate, 1, m.start())

    # Strategy 5: Bold headings (markdown bold)
    for m in _BOLD_HEADING_RE.finditer(text):
        candidate = m.group(1).strip()
        if _is_likely_heading(candidate) and len(candidate) > 3:
            _add(candidate, 2, m.start())

    # Sort by offset
    headings.sort(key=lambda h: h.char_offset)
    return headings


# ---------------------------------------------------------------------------
# Document Structure Builder
# ---------------------------------------------------------------------------

class DocumentStructureBuilder:
    """
    Builds a hierarchical document tree from parsed pages.

    Usage:
        builder = DocumentStructureBuilder()
        tree = builder.build(parse_result, document_name="BSC_300_Manual.pdf")
        sections = builder.flatten_sections(tree)
    """

    def __init__(self):
        self._heading_map: Dict[str, Heading] = {}

    def build(
        self,
        pages: List[Any],
        document_name: str = "unknown",
        document_id: Optional[str] = None,
    ) -> DocumentNode:
        """
        Build hierarchical document tree from parsed pages.

        Args:
            pages: List of ParsePage objects (must have .text and .page_number attributes)
            document_name: Source document filename
            document_id: Optional UUID; generated if not provided

        Returns:
            Root DocumentNode with children representing the full hierarchy.
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        # 1. Extract all headings from all pages
        all_headings: List[Heading] = []
        full_text_parts: List[Tuple[str, int]] = []  # (text, page_num)

        for page in pages:
            page_text = getattr(page, "text", "") or ""
            page_num = getattr(page, "page_number", 0) or 0
            if page_text.strip():
                page_headings = extract_headings(page_text, page_num)
                all_headings.extend(page_headings)
                full_text_parts.append((page_text, page_num))

        if not full_text_parts:
            return DocumentNode(
                node_id=document_id,
                level="document",
                title=document_name,
                content="",
                page_start=0,
                page_end=0,
                parent_id=None,
                heading_path=[document_name],
            )

        # 2. Build concatenated text with page markers for offset tracking
        concat_text = ""
        page_boundaries: List[Tuple[int, int, int]] = []  # (start_offset, end_offset, page_num)
        for page_text, page_num in full_text_parts:
            start = len(concat_text)
            concat_text += page_text + "\n\n"
            end = len(concat_text)
            page_boundaries.append((start, end, page_num))

        # 3. Remap heading offsets to concatenated text
        remapped_headings: List[Tuple[Heading, int]] = []  # (heading, global_offset)
        page_offset_base = 0
        page_idx = 0
        for page_text, page_num in full_text_parts:
            for h in all_headings:
                if h.page_num == page_num:
                    global_offset = page_offset_base + h.char_offset
                    remapped_headings.append((h, global_offset))
            page_offset_base += len(page_text) + 2  # +2 for \n\n

        # Deduplicate and sort
        seen = set()
        unique_headings: List[Tuple[Heading, int]] = []
        for h, offset in sorted(remapped_headings, key=lambda x: x[1]):
            key = (h.text.strip().lower(), h.level)
            if key not in seen:
                seen.add(key)
                unique_headings.append((h, offset))

        # 4. Build the tree using a stack-based approach
        root = DocumentNode(
            node_id=document_id,
            level="document",
            title=document_name,
            content="",
            page_start=full_text_parts[0][1] if full_text_parts else 0,
            page_end=full_text_parts[-1][1] if full_text_parts else 0,
            parent_id=None,
            heading_path=[document_name],
        )

        if not unique_headings:
            # No headings found — entire document is one section
            root.content = concat_text.strip()
            single_section = DocumentNode(
                node_id=str(uuid.uuid4()),
                level="section",
                title=document_name,
                content=concat_text.strip(),
                page_start=root.page_start,
                page_end=root.page_end,
                parent_id=root.node_id,
                heading_path=[document_name],
            )
            root.children.append(single_section)
            return root

        # Build sections from heading boundaries
        level_to_name = {1: "chapter", 2: "section", 3: "subsection", 4: "paragraph"}
        stack: List[DocumentNode] = [root]

        for i, (heading, offset) in enumerate(unique_headings):
            # Determine text range for this section
            next_offset = unique_headings[i + 1][1] if i + 1 < len(unique_headings) else len(concat_text)
            section_text = concat_text[offset:next_offset].strip()

            # Remove the heading line itself from content
            first_newline = section_text.find("\n")
            if first_newline > 0:
                content_text = section_text[first_newline:].strip()
            else:
                content_text = ""

            # Determine page range
            page_start = _offset_to_page(offset, page_boundaries)
            page_end = _offset_to_page(next_offset - 1, page_boundaries)

            level_name = level_to_name.get(heading.level, "subsection")

            # Pop stack until we find appropriate parent
            while len(stack) > 1 and _level_rank(stack[-1].level) >= _level_rank(level_name):
                stack.pop()

            parent = stack[-1]
            heading_path = parent.heading_path + [heading.text.strip()]

            node = DocumentNode(
                node_id=str(uuid.uuid4()),
                level=level_name,
                title=heading.text.strip(),
                content=content_text,
                page_start=page_start,
                page_end=page_end,
                parent_id=parent.node_id,
                heading_path=heading_path,
                metadata={
                    "heading_level": heading.level,
                    "document_id": document_id,
                    "document_name": document_name,
                },
            )

            parent.children.append(node)
            stack.append(node)

        # Handle content before the first heading
        first_heading_offset = unique_headings[0][1] if unique_headings else len(concat_text)
        preamble = concat_text[:first_heading_offset].strip()
        if preamble and len(preamble) > 20:
            preamble_node = DocumentNode(
                node_id=str(uuid.uuid4()),
                level="section",
                title="Introduction",
                content=preamble,
                page_start=root.page_start,
                page_end=_offset_to_page(first_heading_offset, page_boundaries),
                parent_id=root.node_id,
                heading_path=[document_name, "Introduction"],
            )
            root.children.insert(0, preamble_node)

        return root

    def flatten_sections(self, root: DocumentNode) -> List[DocumentNode]:
        """
        Flatten the tree into a list of all section-level nodes (chapters, sections, subsections).
        Excludes the root document node.
        """
        result: List[DocumentNode] = []
        self._flatten_recursive(root, result)
        return result

    def _flatten_recursive(self, node: DocumentNode, result: List[DocumentNode]):
        if node.level != "document":
            result.append(node)
        for child in node.children:
            self._flatten_recursive(child, result)

    def get_section_content(self, node: DocumentNode, include_children: bool = True) -> str:
        """
        Get the full text content of a section, optionally including all child sections.
        """
        parts = [node.content] if node.content else []
        if include_children:
            for child in node.children:
                child_content = self.get_section_content(child, include_children=True)
                if child_content:
                    parts.append(f"\n{child.title}\n{child_content}")
        return "\n\n".join(parts)

    def get_section_summary(self, node: DocumentNode, max_chars: int = 500) -> str:
        """Generate a short summary text for section-level embedding."""
        title = node.title or ""
        path = " > ".join(node.heading_path) if node.heading_path else title
        content = node.content[:max_chars] if node.content else ""
        return f"{path}\n{content}".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_rank(level: str) -> int:
    """Convert level name to numeric rank for tree building."""
    ranks = {"document": 0, "chapter": 1, "section": 2, "subsection": 3, "paragraph": 4}
    return ranks.get(level, 5)


def _offset_to_page(offset: int, page_boundaries: List[Tuple[int, int, int]]) -> int:
    """Convert a character offset in concatenated text to a page number."""
    for start, end, page_num in page_boundaries:
        if start <= offset < end:
            return page_num
    if page_boundaries:
        return page_boundaries[-1][2]
    return 0


def build_document_tree(pages: List[Any], document_name: str = "unknown", document_id: Optional[str] = None) -> DocumentNode:
    """
    Convenience function to build a document tree from parsed pages.

    Args:
        pages: List of ParsePage objects
        document_name: Source filename
        document_id: Optional UUID

    Returns:
        Root DocumentNode
    """
    builder = DocumentStructureBuilder()
    return builder.build(pages, document_name, document_id)
