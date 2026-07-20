"""Document chunker: splits raw text into Chunk objects with metadata."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    position: int


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
    chunks: List["Chunk"],
) -> None:
    """Append a new Chunk to *chunks* if the accumulated text is non-empty."""
    text = " ".join(chunk_parts).strip()
    if text:
        pos = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc_id}_{pos}",
                doc_id=doc_id,
                doc_title=doc_title,
                heading=chunk_heading,
                text=text,
                position=pos,
            )
        )


def _accumulate_paragraph(
    heading: str,
    para_text: str,
    target_chars: int,
    doc_id: str,
    doc_title: str,
    chunk_heading: str,
    chunk_parts: List[str],
    chunk_len: int,
    chunks: List["Chunk"],
) -> tuple:
    """Process one paragraph and return updated (chunk_heading, chunk_parts, chunk_len).

    Emits a chunk to *chunks* when the heading changes or target_chars is exceeded.
    """
    heading_changed = heading != chunk_heading and bool(chunk_parts)
    size_exceeded = chunk_len + len(para_text) > target_chars and bool(chunk_parts)

    if heading_changed:
        _emit_chunk(doc_id, doc_title, chunk_heading, chunk_parts, chunks)
        chunk_parts = []
        chunk_len = 0
        chunk_heading = heading
    elif size_exceeded:
        _emit_chunk(doc_id, doc_title, chunk_heading, chunk_parts, chunks)
        chunk_parts = []
        chunk_len = 0
        # Heading stays the same

    chunk_parts.append(para_text)
    chunk_len += len(para_text)
    return chunk_heading, chunk_parts, chunk_len


def chunk_document(
    doc_id: str,
    raw_text: str,
    target_chars: int = 400,
) -> List[Chunk]:
    """Split raw_text into Chunk objects.

    Detects title (first non-empty line), detects headings via regex,
    splits body into paragraphs separated by blank lines, accumulates
    paragraphs into chunks bounded by target_chars.

    Returns an empty list when raw_text is blank.
    """
    if not raw_text.strip():
        return []

    lines = raw_text.splitlines()
    title = _detect_title(lines, doc_id)

    # Build a list of (heading_or_none, paragraph_text) pairs
    current_heading = ""
    paragraphs: List[tuple] = []  # (heading, text)
    buffer: List[str] = []

    def flush_buffer() -> None:
        text = " ".join(buffer).strip()
        if text:
            paragraphs.append((current_heading, text))

    for line in lines:
        if _is_heading(line):
            flush_buffer()
            buffer = []
            current_heading = _extract_heading_text(line)
        elif line.strip() == "":
            flush_buffer()
            buffer = []
        else:
            buffer.append(line.strip())

    flush_buffer()

    if not paragraphs:
        # Fallback: treat the whole text as one chunk
        return [
            Chunk(
                id=f"{doc_id}_0",
                doc_id=doc_id,
                doc_title=title,
                heading="",
                text=raw_text.strip(),
                position=0,
            )
        ]

    # Accumulate paragraphs into chunks bounded by target_chars
    chunks: List[Chunk] = []
    chunk_heading = paragraphs[0][0]
    chunk_parts: List[str] = []
    chunk_len = 0

    for heading, para_text in paragraphs:
        chunk_heading, chunk_parts, chunk_len = _accumulate_paragraph(
            heading,
            para_text,
            target_chars,
            doc_id,
            title,
            chunk_heading,
            chunk_parts,
            chunk_len,
            chunks,
        )

    _emit_chunk(doc_id, title, chunk_heading, chunk_parts, chunks)

    return chunks
