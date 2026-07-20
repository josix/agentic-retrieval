"""Deterministic, offline contextualizer for RAG chunks.

No model, no network, no randomness.  Output is byte-identical for identical inputs.
"""

import re
from typing import List, Optional

from retrieval.chunker import Chunk


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text (up to 120 chars)."""
    match = re.search(r"[^.!?]*[.!?]", text)
    if match:
        sentence = match.group(0).strip()
    else:
        sentence = text.strip()
    return sentence[:120]


def make_context(chunk: Chunk, neighbors: List[Chunk]) -> str:
    """Build a context prefix for a chunk given its neighbour chunks.

    Format:
        [<doc_title> > <heading>] <heading tokens once> <first sentence of
        previous chunk truncated>. <chunk.text>

    When heading or previous neighbour is absent, degrade gracefully.
    """
    parts: List[str] = []

    # Breadcrumb prefix
    if chunk.heading:
        breadcrumb = f"[{chunk.doc_title} > {chunk.heading}]"
        parts.append(breadcrumb)
        parts.append(chunk.heading)
    else:
        breadcrumb = f"[{chunk.doc_title}]"
        parts.append(breadcrumb)

    # First sentence of the immediately preceding chunk
    prev_chunk: Optional[Chunk] = None
    for nb in neighbors:
        if nb.position == chunk.position - 1 and nb.doc_id == chunk.doc_id:
            prev_chunk = nb
            break

    if prev_chunk:
        prev_sentence = _first_sentence(prev_chunk.text)
        if prev_sentence and not prev_sentence.endswith("."):
            prev_sentence += "."
        parts.append(prev_sentence)

    parts.append(chunk.text)
    return " ".join(parts)


def contextualize(chunk: Chunk, all_chunks_in_doc: List[Chunk]) -> str:
    """Return contextualized text for chunk using all chunks in the same document."""
    return make_context(chunk, all_chunks_in_doc)
