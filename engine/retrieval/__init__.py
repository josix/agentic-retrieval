"""Retrieval package for contextual RAG experiment.

Re-exports the public API used by the plugin's skills and the test suite.
"""

__version__ = "0.7.1"

from retrieval.bm25 import BM25Index
from retrieval.chunker import Chunk, chunk_document
from retrieval.contextualizer import contextualize, make_context
from retrieval.document import Document
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.index import ContextualRetriever, Result
from retrieval.persistence import (
    cached_retrievers,
    compute_fingerprint,
    index_dir,
    is_stale,
    load_index,
    save_index,
)
from retrieval.retrievers import (
    REGISTRY,
    ContextualLexicalRetriever,
    HybridRetriever,
    LexicalRetriever,
    PiSeriniRetriever,
    Retriever,
    TurbovecRetriever,
    build_retriever,
)
from retrieval.tfidf import TfidfIndex, tokenize

__all__ = [
    "Chunk",
    "chunk_document",
    "TfidfIndex",
    "tokenize",
    "BM25Index",
    "reciprocal_rank_fusion",
    "contextualize",
    "make_context",
    "ContextualRetriever",
    "Result",
    "Document",
    "Retriever",
    "LexicalRetriever",
    "ContextualLexicalRetriever",
    "HybridRetriever",
    "TurbovecRetriever",
    "PiSeriniRetriever",
    "REGISTRY",
    "build_retriever",
    "cached_retrievers",
    "compute_fingerprint",
    "index_dir",
    "is_stale",
    "load_index",
    "save_index",
]
