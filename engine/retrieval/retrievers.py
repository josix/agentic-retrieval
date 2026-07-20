"""Pluggable document-level retrievers.

Every retriever implements the same tiny contract:

    name                    -> str            human label for the results table
    index(documents)        -> None           build over a list of Document
    search(query, top_k)    -> List[str]      ranked docids, best first

Four backends:

  * ``LexicalRetriever``  — the project's own BM25 + TF-IDF + RRF, run at the
    document level.  Pure stdlib; always available.
  * ``TurbovecRetriever`` — dense ANN over embeddings via TurboQuant
    (github.com/RyanCodrai/turbovec).  Lazy import; needs the ``turbovec``
    extra plus an embedder.
  * ``PiSeriniRetriever`` — Lucene BM25 via Pyserini, the lexical retriever the
    paper (github.com/justram/pi-serini) reports.  Lazy import; needs the
    ``pyserini`` extra and Java 21.
  * ``HybridRetriever``   — lexical + dense arms indexed together, fused with
    RRF at search time.  Needs whatever ``TurbovecRetriever`` needs.

Optional backends follow the project's stub convention (see
``retrieval/providers.py``): construction may succeed, but the missing
dependency raises a ``RuntimeError`` with opt-in instructions when used.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from retrieval.bm25 import BM25Index
from retrieval.document import Document
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.tfidf import TfidfIndex


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, documents: List[Document]) -> None: ...

    def search(self, query: str, top_k: int) -> List[str]: ...


class LexicalRetriever:
    """Project core: TF-IDF + BM25 fused with RRF, document-level.

    Each document is treated as a single retrieval unit (no sub-chunking), so
    a document's score is the fusion of its TF-IDF and BM25 rankings.
    """

    name = "lexical (bm25+tfidf+rrf)"

    #: Bump when the persisted dict shape changes incompatibly; ``from_dict``
    #: rejects any other value so a stale on-disk cache is rebuilt rather than
    #: mis-parsed.
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._docids: List[str] = []
        self._tfidf = TfidfIndex()
        self._bm25 = BM25Index()

    def index(self, documents: List[Document]) -> None:
        self._docids = [d.docid for d in documents]
        texts = [d.text for d in documents]
        self._tfidf.fit(texts)
        self._bm25.fit(texts)

    def search(self, query: str, top_k: int) -> List[str]:
        tfidf_rank = [idx for idx, _ in self._tfidf.query(query)]
        bm25_rank = [idx for idx, _ in self._bm25.query(query)]
        fused = reciprocal_rank_fusion([tfidf_rank, bm25_rank])
        return [self._docids[idx] for idx, _ in fused[:top_k]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict for on-disk persistence."""
        return {
            "schema": self.SCHEMA_VERSION,
            "docids": self._docids,
            "tfidf": self._tfidf.to_dict(),
            "bm25": self._bm25.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LexicalRetriever":
        """Rebuild a ``LexicalRetriever`` from ``to_dict`` output.

        Raises ``ValueError`` on an unrecognized schema version; callers
        should catch this and reindex from scratch rather than risk mis-
        parsing an incompatible on-disk cache.
        """
        schema = data.get("schema")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported LexicalRetriever schema {schema!r}")
        retriever = cls()
        retriever._docids = data["docids"]
        retriever._tfidf = TfidfIndex.from_dict(data["tfidf"])
        retriever._bm25 = BM25Index.from_dict(data["bm25"])
        return retriever


class ContextualLexicalRetriever(LexicalRetriever):
    """LexicalRetriever over LLM-enriched document text.

    Before indexing, each document's text is prefixed with an LLM-generated
    context (topics + key entities) — the document-granularity analog of
    Anthropic's Contextual Retrieval — then ranked with the same TF-IDF + BM25 +
    RRF as ``LexicalRetriever``.

    Opt-in and online: enrichment is one LLM call per document (cost scales with
    corpus size). Lazy import; needs the ``anthropic`` package +
    ``ANTHROPIC_API_KEY``.
    """

    name = "lexical+llm-context"

    def __init__(self, contextualizer: Optional[Callable[[str], str]] = None) -> None:
        super().__init__()
        self._contextualizer = contextualizer

    def _ensure_contextualizer(self) -> Callable[[str], str]:
        if self._contextualizer is None:
            from retrieval.llm_contextualizer import LLMDocumentContextualizer

            self._contextualizer = LLMDocumentContextualizer().generate
        return self._contextualizer

    def index(self, documents: List[Document]) -> None:
        contextualize = self._ensure_contextualizer()
        enriched = [
            Document(d.docid, f"{contextualize(d.text)} {d.text}".strip(), d.url)
            for d in documents
        ]
        super().index(enriched)


class TurbovecRetriever:
    """Dense ANN comparison via TurboQuant quantization (turbovec).

    Embeds documents with a sentence-transformers model, indexes the vectors
    in a ``TurboQuantIndex``, and ranks by inner-product similarity.  The
    embedder is intentionally pluggable; the default is a small, fast model.
    """

    name = "turbovec (dense ann)"

    #: Bump when the persisted dict shape changes incompatibly (see
    #: ``LexicalRetriever.SCHEMA_VERSION``).
    SCHEMA_VERSION = 1

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bit_width: int = 4) -> None:
        self._model_name = model_name
        self._bit_width = bit_width
        self._docids: List[str] = []
        self._index = None
        self._embedder = None
        self._vectors: Optional[List[List[float]]] = None

    def _require_backends(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            from turbovec import TurboQuantIndex  # type: ignore
        except ImportError as exc:  # pragma: no cover - guidance path
            raise RuntimeError(
                "turbovec retriever needs the 'turbovec' + 'local' extras:\n"
                "  uv pip install -e '.[turbovec,local]'"
            ) from exc
        return SentenceTransformer, TurboQuantIndex

    def index(self, documents: List[Document]) -> None:
        # Check for the optional extras (raises a guidance RuntimeError if
        # missing) before importing numpy, so a missing 'dense' tier always
        # surfaces the documented RuntimeError rather than a raw ImportError.
        SentenceTransformer, TurboQuantIndex = self._require_backends()
        import numpy as np  # provided by the turbovec/local extras

        self._docids = [d.docid for d in documents]
        self._embedder = SentenceTransformer(self._model_name)
        vectors = np.ascontiguousarray(
            self._embedder.encode(
                [d.text for d in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        self._vectors = vectors.tolist()
        self._index = TurboQuantIndex(dim=vectors.shape[1], bit_width=self._bit_width)
        self._index.add(vectors)

    def _ensure_embedder(self):
        # A from_dict-restored retriever has an index but no embedder yet;
        # the query encoder is only needed (and loaded) at search time.
        if self._embedder is None:
            SentenceTransformer, _TurboQuantIndex = self._require_backends()
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    def search(self, query: str, top_k: int) -> List[str]:
        if self._index is None:
            raise RuntimeError("call index() before search()")
        embedder = self._ensure_embedder()
        import numpy as np  # provided by the turbovec/local extras

        # turbovec's kernel takes a 2D float32 batch of queries and returns
        # batched (scores, handles); encode([query]) already yields (1, dim).
        q = np.ascontiguousarray(
            embedder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )
        _scores, handles = self._index.search(q, min(top_k, len(self._docids)))
        return [self._docids[i] for i in handles[0]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (docids + raw embedding vectors).

        The quantized ``TurboQuantIndex`` itself is not serializable, so the
        pre-quantization vectors are stored and the index is rebuilt from
        them in ``from_dict``.
        """
        if self._index is None or self._vectors is None:
            raise RuntimeError("call index() before to_dict()")
        return {
            "schema": self.SCHEMA_VERSION,
            "model_name": self._model_name,
            "bit_width": self._bit_width,
            "docids": self._docids,
            "vectors": self._vectors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurbovecRetriever":
        """Rebuild from ``to_dict`` output; needs the same extras as ``index``.

        Raises ``ValueError`` on an unrecognized schema version and
        ``RuntimeError`` (with opt-in guidance) when the optional backends
        are missing.
        """
        schema = data.get("schema")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported TurbovecRetriever schema {schema!r}")
        retriever = cls(model_name=data["model_name"], bit_width=data["bit_width"])
        _SentenceTransformer, TurboQuantIndex = retriever._require_backends()
        import numpy as np  # provided by the turbovec/local extras

        vectors = np.ascontiguousarray(np.asarray(data["vectors"], dtype=np.float32))
        retriever._docids = data["docids"]
        retriever._vectors = data["vectors"]
        retriever._index = TurboQuantIndex(dim=vectors.shape[1], bit_width=retriever._bit_width)
        retriever._index.add(vectors)
        return retriever


class PiSeriniRetriever:
    """Lucene BM25 via Pyserini — the paper's reference lexical retriever.

    Builds an in-memory Lucene index over the corpus and queries it with
    Pyserini's ``LuceneSearcher``.  Requires Java 21 (Pyserini wraps Anserini).
    """

    name = "pi-serini (lucene bm25)"

    #: Bump when the persisted dict shape changes incompatibly (see
    #: ``LexicalRetriever.SCHEMA_VERSION``).
    SCHEMA_VERSION = 1

    def __init__(self, k1: float = 0.9, b: float = 0.4,
                 index_path: "Optional[Path | str]" = None) -> None:
        self._k1 = k1
        self._b = b
        self._searcher = None
        # With an explicit index_path the Lucene index survives the process
        # (and can be reopened via from_dict); without one it lives in a
        # throwaway tempdir, matching the original in-memory-style behavior.
        self._index_dir = str(index_path) if index_path is not None else None
        self._docids: List[str] = []

    def _require_backend(self):
        try:
            from pyserini.index.lucene import LuceneIndexer  # type: ignore
            from pyserini.search.lucene import LuceneSearcher  # type: ignore
        except ImportError as exc:  # pragma: no cover - guidance path
            raise RuntimeError(
                "pi-serini retriever needs the 'pyserini' extra and Java 21:\n"
                "  uv pip install -e '.[pyserini]'   (and install a JDK 21)"
            ) from exc
        return LuceneIndexer, LuceneSearcher

    def index(self, documents: List[Document]) -> None:
        import shutil
        import tempfile

        LuceneIndexer, LuceneSearcher = self._require_backend()
        if self._index_dir is None:
            self._index_dir = tempfile.mkdtemp(prefix="piserini_bcp_")
        else:
            # Lucene appends to an existing index; clear the target so a
            # rebuild never mixes stale segments with fresh ones.
            shutil.rmtree(self._index_dir, ignore_errors=True)
            Path(self._index_dir).mkdir(parents=True, exist_ok=True)
        indexer = LuceneIndexer(self._index_dir)
        indexer.add_batch_dict(
            [{"id": d.docid, "contents": d.text} for d in documents]
        )
        indexer.close()
        self._docids = [d.docid for d in documents]
        self._searcher = LuceneSearcher(self._index_dir)
        self._searcher.set_bm25(self._k1, self._b)

    def search(self, query: str, top_k: int) -> List[str]:
        if self._searcher is None:
            raise RuntimeError("call index() before search()")
        hits = self._searcher.search(query, k=top_k)
        return [hit.docid for hit in hits]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe pointer at the on-disk Lucene index.

        The Lucene segments themselves stay in ``index_path`` (they are
        binary and already on disk); only the path + BM25 params + docids
        are stored, so persistence is only meaningful when the retriever
        was built with an explicit, durable ``index_path``.
        """
        if self._searcher is None:
            raise RuntimeError("call index() before to_dict()")
        return {
            "schema": self.SCHEMA_VERSION,
            "k1": self._k1,
            "b": self._b,
            "index_dir": self._index_dir,
            "docids": self._docids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PiSeriniRetriever":
        """Reopen a searcher over the persisted Lucene index directory.

        Raises ``ValueError`` on an unrecognized schema version or a missing
        index directory (callers should treat that as "no usable cache" and
        reindex), and ``RuntimeError`` when the ``pyserini`` extra is absent.
        """
        schema = data.get("schema")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported PiSeriniRetriever schema {schema!r}")
        index_dir = data["index_dir"]
        if not index_dir or not Path(index_dir).is_dir():
            raise ValueError(f"pi-serini lucene index dir missing: {index_dir!r}")
        retriever = cls(k1=data["k1"], b=data["b"], index_path=index_dir)
        _LuceneIndexer, LuceneSearcher = retriever._require_backend()
        retriever._docids = data["docids"]
        retriever._searcher = LuceneSearcher(index_dir)
        retriever._searcher.set_bm25(retriever._k1, retriever._b)
        return retriever


class HybridRetriever:
    """Lexical + dense arms over the same corpus, fused with RRF at search time.

    Indexes both a ``LexicalRetriever`` and a ``TurbovecRetriever`` over the
    documents; ``search`` takes each arm's ranking, remaps docids into a
    shared integer space, and fuses with ``reciprocal_rank_fusion``.  Needs
    the same optional extras as ``TurbovecRetriever`` (indexing raises their
    guidance ``RuntimeError`` when absent).
    """

    name = "hybrid (lexical+turbovec rrf)"

    #: Bump when the persisted dict shape changes incompatibly (see
    #: ``LexicalRetriever.SCHEMA_VERSION``).
    SCHEMA_VERSION = 1

    def __init__(self, dense: Optional[TurbovecRetriever] = None) -> None:
        self._lexical = LexicalRetriever()
        self._dense = dense if dense is not None else TurbovecRetriever()

    def index(self, documents: List[Document]) -> None:
        # Dense arm first: it fails fast (with opt-in guidance) when the
        # turbovec extras are missing, before any lexical work is done.
        self._dense.index(documents)
        self._lexical.index(documents)

    def search(self, query: str, top_k: int) -> List[str]:
        # Pull a deeper candidate pool from each arm than the caller asked
        # for, so RRF has overlap to work with before truncating to top_k.
        pool = max(top_k * 3, 10)
        lexical_docids = self._lexical.search(query, pool)
        dense_docids = self._dense.search(query, pool)
        all_docids = sorted(set(lexical_docids) | set(dense_docids))
        to_idx = {docid: i for i, docid in enumerate(all_docids)}
        fused = reciprocal_rank_fusion([
            [to_idx[d] for d in lexical_docids],
            [to_idx[d] for d in dense_docids],
        ])
        return [all_docids[idx] for idx, _score in fused[:top_k]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize both arms to a JSON-safe dict for on-disk persistence."""
        lexical_data = self._lexical.to_dict()
        return {
            "schema": self.SCHEMA_VERSION,
            "docids": lexical_data["docids"],
            "lexical": lexical_data,
            "dense": self._dense.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridRetriever":
        """Rebuild both arms from ``to_dict`` output.

        Raises ``ValueError`` on an unrecognized schema version and
        ``RuntimeError`` when the dense arm's extras are missing.
        """
        schema = data.get("schema")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported HybridRetriever schema {schema!r}")
        retriever = cls()
        retriever._lexical = LexicalRetriever.from_dict(data["lexical"])
        retriever._dense = TurbovecRetriever.from_dict(data["dense"])
        return retriever


# Registry keyed by CLI name.
REGISTRY = {
    "lexical": LexicalRetriever,
    "lexical+ctx": ContextualLexicalRetriever,
    "turbovec": TurbovecRetriever,
    "pi-serini": PiSeriniRetriever,
    "hybrid": HybridRetriever,
}


def build_retriever(name: str) -> Retriever:
    if name not in REGISTRY:
        raise ValueError(f"unknown retriever {name!r}; choose from {list(REGISTRY)}")
    return REGISTRY[name]()
