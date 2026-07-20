"""Minimal document record shared by the retriever adapters.

Pure stdlib, zero dependencies — kept separate from ``retrieval/index.py``
(which deals in ``Chunk``s) because the retrievers in ``retrieval/retrievers.py``
operate at document granularity.
"""

from dataclasses import dataclass


@dataclass
class Document:
    docid: str
    text: str
    url: str = ""
