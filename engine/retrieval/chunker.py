"""Document chunker: splits raw text into Chunk objects with metadata."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

# ast_chunker.py imports no optional (tree-sitter) deps at module scope —
# its tree-sitter-language-pack import is lazy, inside chunk_code() — so
# importing LANGUAGE_BY_SUFFIX here is always safe, even without the
# treesitter extra installed (see retrieval/project_loader.py, which does
# the same top-level import).
from retrieval.ast_chunker import LANGUAGE_BY_SUFFIX

#: File suffixes chunked as prose (``ChunkingPolicy.prose_chars``).
PROSE_SUFFIXES = frozenset({".md", ".markdown", ".rst", ".txt"})

#: File suffixes chunked as structured config (``ChunkingPolicy.config_chars``).
CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"})


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    position: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ChunkingPolicy:
    """Per-file-kind chunk-size policy, agent-decidable via ``retrieval
    tune``/``--auto`` (see ``retrieval.autotune``).

    All five sizes default to ``400`` so ``DEFAULT_POLICY`` reproduces
    today's chunking behavior exactly (the historical single ``target_chars``
    parameter); larger values (e.g. ``code_chars=1200``, ``config_chars=700``)
    only come from autotune's signal-based heuristics or an explicit
    override, never from this default.
    """

    prose_chars: int = 400
    config_chars: int = 400
    code_chars: int = 400
    default_chars: int = 400
    ast_max_chars: int = 1200

    def target_chars_for(self, path: str) -> int:
        """Return the line-chunker ``target_chars`` bucket for *path*, by
        suffix: prose -> ``prose_chars``, structured config ->
        ``config_chars``, a tree-sitter-mapped code suffix -> ``code_chars``,
        anything else -> ``default_chars``."""
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in PROSE_SUFFIXES:
            return self.prose_chars
        if suffix in CONFIG_SUFFIXES:
            return self.config_chars
        if suffix in LANGUAGE_BY_SUFFIX:
            return self.code_chars
        return self.default_chars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prose_chars": self.prose_chars,
            "config_chars": self.config_chars,
            "code_chars": self.code_chars,
            "default_chars": self.default_chars,
            "ast_max_chars": self.ast_max_chars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingPolicy":
        fields = ("prose_chars", "config_chars", "code_chars", "default_chars", "ast_max_chars")
        return cls(**{k: data[k] for k in fields if k in data})


#: The historical, single-``target_chars=400`` behavior — every bucket at
#: 400, so passing ``policy=None`` anywhere reproduces today's chunking.
DEFAULT_POLICY = ChunkingPolicy()


def _detect_title(lines: List[str], doc_id: str) -> str:
    """Return the first non-empty line as the document title."""
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return doc_id


def _is_heading(line: str) -> bool:
    """Return True if the line looks like a section heading."""
    stripped = line.strip()
    # Markdown headings: # Title or ## Title
    if re.match(r"^#{1,6}\s+\S", stripped):
        return True
    # UPPERCASE: style headings (e.g. "Routing:", "Overview:")
    if re.match(r"^[A-Z][A-Za-z ]+:$", stripped):
        return True
    return False


def _extract_heading_text(line: str) -> str:
    """Strip heading markers and return clean heading text."""
    stripped = line.strip()
    # Remove leading # characters
    match = re.match(r"^#{1,6}\s+(.+)$", stripped)
    if match:
        return match.group(1).strip()
    # Remove trailing colon from UPPERCASE: style
    if stripped.endswith(":"):
        return stripped[:-1].strip()
    return stripped


def _emit_chunk(
    doc_id: str,
    doc_title: str,
    chunk_heading: str,
    chunk_parts: List[str],
    chunk_start: Optional[int],
    chunk_end: Optional[int],
    chunk_heading_line: Optional[int],
    chunks: List["Chunk"],
) -> None:
    """Append a new Chunk to *chunks* if the accumulated text is non-empty.

    The span is the paragraphs' min start / max end line, extended to also
    cover *chunk_heading_line* — the line of the heading that introduced this
    chunk's section — so a chunk that opens a new heading includes that
    heading line in its ``[start_line, end_line]`` span.
    """
    text = " ".join(chunk_parts).strip()
    if text:
        pos = len(chunks)
        start_line = chunk_start
        if chunk_heading_line is not None and (
            start_line is None or chunk_heading_line < start_line
        ):
            start_line = chunk_heading_line
        chunks.append(
            Chunk(
                id=f"{doc_id}_{pos}",
                doc_id=doc_id,
                doc_title=doc_title,
                heading=chunk_heading,
                text=text,
                position=pos,
                start_line=start_line if start_line is not None else 1,
                end_line=chunk_end if chunk_end is not None else start_line or 1,
            )
        )


def _accumulate_paragraph(
    heading: str,
    para_text: str,
    para_start: int,
    para_end: int,
    heading_line: Optional[int],
    target_chars: int,
    doc_id: str,
    doc_title: str,
    chunk_heading: str,
    chunk_parts: List[str],
    chunk_len: int,
    chunk_start: Optional[int],
    chunk_end: Optional[int],
    chunk_heading_line: Optional[int],
    chunks: List["Chunk"],
) -> tuple:
    """Process one paragraph and return updated chunk-accumulation state.

    Returns ``(chunk_heading, chunk_parts, chunk_len, chunk_start, chunk_end,
    chunk_heading_line)``. Emits a chunk to *chunks* when the heading changes
    or target_chars is exceeded.
    """
    heading_changed = heading != chunk_heading and bool(chunk_parts)
    size_exceeded = chunk_len + len(para_text) > target_chars and bool(chunk_parts)

    if heading_changed:
        _emit_chunk(
            doc_id, doc_title, chunk_heading, chunk_parts,
            chunk_start, chunk_end, chunk_heading_line, chunks,
        )
        chunk_parts = []
        chunk_len = 0
        chunk_start = None
        chunk_end = None
        chunk_heading = heading
        chunk_heading_line = heading_line
    elif size_exceeded:
        _emit_chunk(
            doc_id, doc_title, chunk_heading, chunk_parts,
            chunk_start, chunk_end, chunk_heading_line, chunks,
        )
        chunk_parts = []
        chunk_len = 0
        chunk_start = None
        chunk_end = None
        # Heading stays the same, but its introducing line was already
        # captured by the previous chunk — don't re-extend this one.
        chunk_heading_line = None

    chunk_parts.append(para_text)
    chunk_len += len(para_text)
    if chunk_start is None or para_start < chunk_start:
        chunk_start = para_start
    if chunk_end is None or para_end > chunk_end:
        chunk_end = para_end
    return chunk_heading, chunk_parts, chunk_len, chunk_start, chunk_end, chunk_heading_line


def chunk_document(
    doc_id: str,
    raw_text: str,
    target_chars: int = 400,
) -> List[Chunk]:
    """Split raw_text into Chunk objects.

    Detects title (first non-empty line), detects headings via regex,
    splits body into paragraphs separated by blank lines, accumulates
    paragraphs into chunks bounded by target_chars. Every chunk carries a
    1-based ``[start_line, end_line]`` span over *raw_text*'s source lines,
    so results can be turned into ``Read(path, offset=start_line,
    limit=end_line-start_line+1)`` calls.

    Returns an empty list when raw_text is blank.
    """
    if not raw_text.strip():
        return []

    lines = raw_text.splitlines()
    title = _detect_title(lines, doc_id)

    # Build a list of (heading, text, start_line, end_line, heading_line) tuples.
    current_heading = ""
    current_heading_line: Optional[int] = None
    paragraphs: List[Tuple[str, str, int, int, Optional[int]]] = []
    buffer: List[str] = []
    buffer_start: Optional[int] = None
    buffer_end: Optional[int] = None

    def flush_buffer() -> None:
        nonlocal buffer, buffer_start, buffer_end
        text = " ".join(buffer).strip()
        if text and buffer_start is not None and buffer_end is not None:
            paragraphs.append(
                (current_heading, text, buffer_start, buffer_end, current_heading_line)
            )
        buffer = []
        buffer_start = None
        buffer_end = None

    for line_num, line in enumerate(lines, start=1):
        if _is_heading(line):
            flush_buffer()
            current_heading = _extract_heading_text(line)
            current_heading_line = line_num
        elif line.strip() == "":
            flush_buffer()
        else:
            if buffer_start is None:
                buffer_start = line_num
            buffer_end = line_num
            buffer.append(line.strip())

    flush_buffer()

    if not paragraphs:
        # Fallback: treat the whole text as one chunk spanning every line.
        return [
            Chunk(
                id=f"{doc_id}_0",
                doc_id=doc_id,
                doc_title=title,
                heading="",
                text=raw_text.strip(),
                position=0,
                start_line=1,
                end_line=len(lines) if lines else 1,
            )
        ]

    # Accumulate paragraphs into chunks bounded by target_chars
    chunks: List[Chunk] = []
    chunk_heading = paragraphs[0][0]
    chunk_parts: List[str] = []
    chunk_len = 0
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None
    chunk_heading_line: Optional[int] = paragraphs[0][4]

    for heading, para_text, para_start, para_end, heading_line in paragraphs:
        (
            chunk_heading,
            chunk_parts,
            chunk_len,
            chunk_start,
            chunk_end,
            chunk_heading_line,
        ) = _accumulate_paragraph(
            heading,
            para_text,
            para_start,
            para_end,
            heading_line,
            target_chars,
            doc_id,
            title,
            chunk_heading,
            chunk_parts,
            chunk_len,
            chunk_start,
            chunk_end,
            chunk_heading_line,
            chunks,
        )

    _emit_chunk(
        doc_id, title, chunk_heading, chunk_parts,
        chunk_start, chunk_end, chunk_heading_line, chunks,
    )

    return chunks
