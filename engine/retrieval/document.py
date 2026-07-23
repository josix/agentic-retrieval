"""Minimal document record shared by the retriever adapters.

Pure stdlib, zero dependencies — kept separate from ``retrieval/index.py``
(which deals in ``Chunk``s) because the retrievers in ``retrieval/retrievers.py``
operate at document granularity.

``Document`` now also carries chunk-span metadata (``source_path``,
``start_line``, ``end_line``) so a chunk-granularity ``Document`` (see
``retrieval.project_loader.load_chunk_documents``) still round-trips through
document-level retrievers without losing the file:line span a coding agent
needs to ``Read`` the exact content. Whole-file callers leave the span
fields at their defaults, preserving backward compatibility.

``context`` is an optional breadcrumb (e.g. ``"Bar.baz"`` for a method
``baz`` nested in class ``Bar``) set by AST-boundary chunking (see
``retrieval.ast_chunker``); it defaults to ``""`` for every other loader,
preserving backward-compatible positional construction.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    docid: str
    text: str
    url: str = ""
    source_path: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    context: str = ""


@dataclass
class SearchHit:
    """One ranked result: a chunk docid plus its file:line span."""

    docid: str
    source_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    rank: int
    context: str = ""
