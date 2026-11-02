"""Pluggable document-level retrievers.

Every retriever implements the same tiny contract:

    name                            -> str             human label for the results table
    index(documents)                -> None             build over a list of Document
    search(query, top_k)            -> List[str]        ranked docids, best first
    search_detailed(query, top_k)   -> List[SearchHit]   ranked hits with file:line spans

``documents`` are expected to be chunk-granularity (see
``retrieval.project_loader.load_chunk_documents``): each carries a
``docid`` of the form ``"{path}:{start}-{end}"`` plus the same span as
structured ``source_path``/``start_line``/``end_line`` fields. Every
retriever tracks a parallel ``self._units`` list (``{docid, source_path,
start_line, end_line}``) built straight from the indexed Documents'
metadata — never by parsing spans back out of a docid string — and
``search_detailed`` maps each ranked positional index into its unit to
build a ``SearchHit``. ``search()`` is a thin wrapper: ``[h.docid for h in
search_detailed(...)]``, kept for backward-compatible callers.

Six backends:

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
  * ``TreeSitterRetriever`` — the same lexical (BM25+TF-IDF+RRF) ranking over
    AST-boundary chunks (see ``retrieval.ast_chunker``), enriched with a
    breadcrumb ``context`` (enclosing function/class) prefixed into the
    ranked text and carried onto each hit.  Tree-sitter is only needed at
    chunking time (in ``retrieval.project_loader.load_ast_chunk_documents``),
    so the retriever itself has no optional deps.

Optional backends follow the project's stub convention (see
``retrieval/providers.py``): construction may succeed, but the missing
dependency raises a ``RuntimeError`` with opt-in instructions when used.
"""

import base64
import hashlib
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from retrieval.bm25 import BM25Index
from retrieval.document import Document, SearchHit
from retrieval.fusion import candidate_pool, reciprocal_rank_fusion
from retrieval.tfidf import TfidfIndex


def _units_from_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """Build the ``{docid, source_path, start_line, end_line}`` unit list a
    retriever tracks alongside its index, straight from each Document's span
    metadata (never parsed back out of the docid string)."""
    units = []
    for d in documents:
        unit: Dict[str, Any] = {
            "docid": d.docid,
            "source_path": d.source_path,
            "start_line": d.start_line,
            "end_line": d.end_line,
            "context": d.context,
        }
        if d.metadata:
            unit["metadata"] = dict(d.metadata)
        units.append(unit)
    return units


def _hits_from_units(units: List[Dict[str, Any]], ranked_idx: List[int]) -> List[SearchHit]:
    """Map ranked positional indices into *units* to build ``SearchHit``s."""
    return [
        SearchHit(
            docid=units[idx]["docid"],
            source_path=units[idx]["source_path"],
            start_line=units[idx]["start_line"],
            end_line=units[idx]["end_line"],
            rank=rank,
            context=units[idx].get("context", ""),
            metadata=dict(units[idx].get("metadata") or {}),
        )
        for rank, idx in enumerate(ranked_idx)
    ]


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, documents: List[Document]) -> None: ...

    def search(self, query: str, top_k: int) -> List[str]: ...

    def search_detailed(self, query: str, top_k: int) -> List[SearchHit]: ...


_MISSING = object()


class LexicalRetriever:
    """Project core: TF-IDF + BM25 fused with RRF, document-level.

    Each document is treated as a single retrieval unit (no sub-chunking), so
    a document's score is the fusion of its TF-IDF and BM25 rankings.
    """

    name = "lexical (bm25+tfidf+rrf)"

    #: Hyperparameter keys this retriever's constructor accepts (see
    #: ``build_retriever`` and ``retrieval.persistence.relevant_params``).
    ACCEPTS = frozenset({"bm25_k1", "bm25_b", "tokenizer"})

    #: Bump when the persisted dict shape changes incompatibly; ``from_dict``
    #: rejects any other value so a stale on-disk cache is rebuilt rather than
    #: mis-parsed. v2 adds ``units`` (chunk span metadata). v3 adds a
    #: persisted ``tokenizer`` mode on the nested tfidf/bm25 dicts, restored
    #: (not re-derived) at query time.
    SCHEMA_VERSION = 3

    #: Importable modules needed to load this retriever's cache (mirrors the
    #: ``_require_backend(s)`` checks) — lets freshness checks avoid a full load.
    REQUIRED_MODULES: tuple = ()

    def __init__(
        self, *, bm25_k1: float = 1.5, bm25_b: float = 0.75, tokenizer: str = "plain"
    ) -> None:
        self._docids: List[str] = []
        self._units: List[Dict[str, Any]] = []
        self._params: Dict[str, Any] = {
            "bm25_k1": bm25_k1, "bm25_b": bm25_b, "tokenizer": tokenizer,
        }
        self._tfidf = TfidfIndex(tokenizer=tokenizer)
        self._bm25 = BM25Index(k1=bm25_k1, b=bm25_b, tokenizer=tokenizer)

    @property
    def doc_count(self) -> int:
        """Number of indexed chunks (== ``len(to_dict()["docids"])``)."""
        return len(self._docids)

    def index(self, documents: List[Document]) -> None:
        self._units = _units_from_documents(documents)
        self._docids = [u["docid"] for u in self._units]
        texts = [d.text for d in documents]
        self._tfidf.fit(texts)
        self._bm25.fit(texts)

    def search_detailed(self, query: str, top_k: int) -> List[SearchHit]:
        tfidf_rank = [idx for idx, _ in self._tfidf.query(query)]
        bm25_rank = [idx for idx, _ in self._bm25.query(query)]
        fused = reciprocal_rank_fusion([tfidf_rank, bm25_rank])
        ranked_idx = [idx for idx, _ in fused[:top_k]]
        return _hits_from_units(self._units, ranked_idx)

    def search(self, query: str, top_k: int) -> List[str]:
        return [h.docid for h in self.search_detailed(query, top_k)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict for on-disk persistence."""
        return {
            "schema": self.SCHEMA_VERSION,
            "docids": self._docids,
            "units": self._units,
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
        bm25_data = data["bm25"]
        retriever = cls(
            bm25_k1=bm25_data["k1"],
            bm25_b=bm25_data["b"],
            tokenizer=bm25_data.get("tokenizer", "plain"),
        )
        retriever._docids = data["docids"]
        retriever._units = data["units"]
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

    def __init__(
        self, contextualizer: Optional[Callable[[str], str]] = None, **params: Any
    ) -> None:
        super().__init__(**params)
        self._contextualizer = contextualizer

    def _ensure_contextualizer(self) -> Callable[[str], str]:
        if self._contextualizer is None:
            from retrieval.llm_contextualizer import LLMDocumentContextualizer

            self._contextualizer = LLMDocumentContextualizer().generate
        return self._contextualizer

    def index(self, documents: List[Document]) -> None:
        # Enrichment only ever alters `text` (prepending LLM-generated
        # context); span metadata (source_path/start_line/end_line) and the
        # document's own `context` breadcrumb are carried through unchanged
        # so results still resolve to the original file:line location.
        contextualize = self._ensure_contextualizer()
        enriched = [
            Document(
                docid=d.docid,
                text=f"{contextualize(d.text)} {d.text}".strip(),
                url=d.url,
                source_path=d.source_path,
                start_line=d.start_line,
                end_line=d.end_line,
                context=d.context,
                metadata=d.metadata,
            )
            for d in documents
        ]
        super().index(enriched)


class TreeSitterRetriever(LexicalRetriever):
    """LexicalRetriever over AST-boundary ("cAST") chunked documents.

    Expects documents chunk-granularity via
    ``retrieval.project_loader.load_ast_chunk_documents``, each optionally
    carrying a ``context`` breadcrumb (enclosing function/class path, e.g.
    ``"Bar.baz"``). Before indexing, each document's breadcrumb is prefixed
    into its ranked text (so a query for "Bar baz" can match a chunk purely
    via its enclosing-scope name), then ranked with the same TF-IDF + BM25 +
    RRF as ``LexicalRetriever``. Tree-sitter itself is only needed at
    chunking time, not here, so this retriever has zero optional deps.
    """

    name = "tree-sitter (ast chunks)"

    def index(self, documents: List[Document]) -> None:
        enriched = [
            Document(
                docid=d.docid,
                text=f"{d.context}\n{d.text}".strip() if d.context else d.text,
                url=d.url,
                source_path=d.source_path,
                start_line=d.start_line,
                end_line=d.end_line,
                context=d.context,
                metadata=d.metadata,
            )
            for d in documents
        ]
        super().index(enriched)


#: What a seed provider may return: one payload, ``None``, or a lazy iterable
#: of candidate payloads tried in order until one is compatible.
_SeedResult = Union[None, Dict[str, Any], Iterable[Dict[str, Any]]]


class TurbovecRetriever:
    """Dense ANN comparison via TurboQuant quantization (turbovec).

    Embeds documents with a sentence-transformers model, indexes the vectors
    in a ``TurboQuantIndex``, and ranks by inner-product similarity.  The
    embedder is intentionally pluggable; the default is a small, fast model.

    Embeddings are reused across rebuilds: each chunk is keyed by
    ``(model_name, sha256(chunk text))``, and when a seed (the previous
    cache, see ``set_embedding_seed``) holds that key its vector is copied
    byte-for-byte instead of re-encoded. Only unseen texts are encoded, and
    duplicate texts are encoded once.
    """

    name = "turbovec (dense ann)"

    #: Hyperparameter keys this retriever's constructor accepts (see
    #: ``build_retriever`` and ``retrieval.persistence.relevant_params``).
    ACCEPTS = frozenset({"model_name", "bit_width"})

    #: Bump when the persisted dict shape changes incompatibly (see
    #: ``LexicalRetriever.SCHEMA_VERSION``). v2 adds ``units`` (chunk span
    #: metadata). v3 stores vectors as base64 little-endian float32 plus a
    #: per-chunk ``text_sha256`` list enabling embedding reuse.
    SCHEMA_VERSION = 3

    #: Schemas ``from_dict`` accepts. v2 (JSON float lists, no hashes) still
    #: loads and serves queries, but gives no embedding reuse; the next
    #: rebuild writes v3.
    LOADABLE_SCHEMAS = frozenset({2, 3})

    #: See ``LexicalRetriever.REQUIRED_MODULES``.
    REQUIRED_MODULES: tuple = ("sentence_transformers", "turbovec", "numpy")

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bit_width: int = 4) -> None:
        self._model_name = model_name
        self._bit_width = bit_width
        self._docids: List[str] = []
        self._units: List[Dict[str, Any]] = []
        self._index = None
        self._embedder = None
        #: float32 ``(n, dim)`` ndarray (pre-quantization), never a list.
        self._vectors: Optional[Any] = None
        self._text_hashes: Optional[List[str]] = None
        self._seed_provider: Optional[Callable[[], _SeedResult]] = None
        self._reuse_stats: Dict[str, int] = {}

    @property
    def doc_count(self) -> int:
        """Number of indexed chunks (== ``len(to_dict()["docids"])``)."""
        return len(self._docids)

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

    # -- vector (de)serialization -------------------------------------------

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _encode_vectors(arr: Any) -> Dict[str, Any]:
        """float32 ``(n, dim)`` ndarray -> JSON-safe base64 little-endian dict."""
        import numpy as np  # provided by the turbovec/local extras

        arr = np.ascontiguousarray(arr, dtype="<f4")
        return {
            "encoding": "base64",
            "dtype": "<f4",
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        }

    @staticmethod
    def _decode_vectors(payload: Any) -> Any:
        """Decode a v3 base64 dict (or a v2 nested float list) into an owned,
        writable, C-contiguous float32 ``(n, dim)`` ndarray. Raises
        ``ValueError`` on any malformed payload."""
        import numpy as np  # provided by the turbovec/local extras

        if isinstance(payload, dict):
            if payload.get("encoding") != "base64" or payload.get("dtype") != "<f4":
                raise ValueError("unsupported turbovec vector encoding")
            shape = payload.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not all(isinstance(x, int) and x >= 0 for x in shape)
            ):
                raise ValueError("malformed turbovec vector shape")
            encoded = payload.get("data")
            if not isinstance(encoded, str):
                raise ValueError("turbovec vector data must be a base64 string")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != shape[0] * shape[1] * 4:
                raise ValueError("turbovec vector byte length does not match shape")
            return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=True)
        arr = np.ascontiguousarray(np.asarray(payload, dtype=np.float32))
        if arr.ndim != 2:
            raise ValueError("turbovec vectors must be 2-dimensional")
        return arr

    def set_embedding_seed(
        self, provider: Optional[Callable[[], _SeedResult]]
    ) -> None:
        """Register a zero-arg callable returning a prior ``to_dict`` payload
        (or ``None``). It is invoked lazily, only inside ``index()``, so the
        previous cache is read before it is overwritten."""
        self._seed_provider = provider

    def _decode_one_seed(self, data: Any) -> Optional[Dict[str, Any]]:
        """Decode one candidate payload, or ``None`` if unusable here."""
        if not isinstance(data, dict) or data.get("model_name") != self._model_name:
            return None
        hashes = data.get("text_sha256")
        if not hashes:
            return None
        vectors = self._decode_vectors(data["vectors"])
        if vectors.shape[0] != len(hashes):
            return None
        rows: Dict[str, int] = {}
        for row, h in enumerate(hashes):
            rows.setdefault(h, row)
        return {"rows": rows, "vectors": vectors}

    def _decode_seed(self) -> Optional[Dict[str, Any]]:
        """``{"rows": {hash: row}, "vectors": ndarray}`` from the first
        compatible seed candidate, or ``None``. Never raises."""
        if self._seed_provider is None:
            return None
        try:
            result = self._seed_provider()
            candidates = [result] if isinstance(result, dict) or result is None else result
            for data in candidates:
                try:
                    seed = self._decode_one_seed(data)
                except Exception:  # a bad candidate: try the next one
                    continue
                if seed is not None:
                    return seed
        except Exception:  # a bad seed must never fail a build
            pass
        return None

    def _encode(self, texts: List[str]) -> Any:
        import numpy as np  # provided by the turbovec/local extras

        return np.ascontiguousarray(
            self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    @staticmethod
    def _partition_by_seed(
        documents: List[Document], hashes: List[str], seed_rows: Dict[str, int]
    ) -> Tuple[List[int], List[int], List[int], List[int], List[str]]:
        """Split into seed hits and misses (misses de-duplicated by hash).

        Returns ``hit_pos, hit_rows, miss_pos, miss_uidx, miss_texts``.
        """
        hit_pos: List[int] = []
        hit_rows: List[int] = []
        miss_pos: List[int] = []
        miss_uidx: List[int] = []
        miss_texts: List[str] = []
        unique_of: Dict[str, int] = {}
        for pos, (doc, h) in enumerate(zip(documents, hashes)):
            row = seed_rows.get(h)
            if row is not None:
                hit_pos.append(pos)
                hit_rows.append(row)
                continue
            u = unique_of.get(h)
            if u is None:
                u = unique_of[h] = len(miss_texts)
                miss_texts.append(doc.text)
            miss_pos.append(pos)
            miss_uidx.append(u)
        return hit_pos, hit_rows, miss_pos, miss_uidx, miss_texts

    def _encode_all(
        self, documents: List[Document], hashes: List[str]
    ) -> Tuple[Any, List[int], List[str]]:
        """Fallback (seed unusable): encode every unique text once.

        Returns ``enc, miss_uidx, miss_texts``.
        """
        miss_uidx: List[int] = []
        miss_texts: List[str] = []
        first: Dict[str, int] = {}
        for h, doc in zip(hashes, documents):
            if h not in first:
                first[h] = len(miss_texts)
                miss_texts.append(doc.text)
            miss_uidx.append(first[h])
        return self._encode(miss_texts), miss_uidx, miss_texts

    def _index_empty(self, SentenceTransformer: Any, TurboQuantIndex: Any) -> None:
        self._embedder = SentenceTransformer(self._model_name)
        vectors = self._encode([])
        self._vectors = vectors
        self._text_hashes = []
        self._reuse_stats = {"reused": 0, "encoded": 0}
        self._index = TurboQuantIndex(dim=vectors.shape[1], bit_width=self._bit_width)
        self._index.add(vectors)

    def index(self, documents: List[Document]) -> None:
        # Check for the optional extras (raises a guidance RuntimeError if
        # missing) before importing numpy, so a missing 'dense' tier always
        # surfaces the documented RuntimeError rather than a raw ImportError.
        SentenceTransformer, TurboQuantIndex = self._require_backends()
        import numpy as np  # provided by the turbovec/local extras

        self._units = _units_from_documents(documents)
        self._docids = [u["docid"] for u in self._units]
        if not documents:
            self._index_empty(SentenceTransformer, TurboQuantIndex)
            return

        hashes = [self._text_hash(d.text) for d in documents]
        seed = self._decode_seed()
        hit_pos, hit_rows, miss_pos, miss_uidx, miss_texts = self._partition_by_seed(
            documents, hashes, seed["rows"] if seed else {}
        )

        enc = None
        if miss_texts:
            self._embedder = SentenceTransformer(self._model_name)
            enc = self._encode(miss_texts)
        if seed is not None and enc is not None and enc.shape[1] != seed["vectors"].shape[1]:
            # Dimension drift: the seed is unusable; encode everything.
            seed, hit_pos, hit_rows = None, [], []
            miss_pos = list(range(len(documents)))
            enc, miss_uidx, miss_texts = self._encode_all(documents, hashes)
        dim = enc.shape[1] if enc is not None else seed["vectors"].shape[1]

        out = np.empty((len(documents), dim), dtype=np.float32)
        if hit_pos:
            out[hit_pos] = seed["vectors"][hit_rows]
        if miss_pos:
            out[miss_pos] = enc[miss_uidx]

        self._vectors = out
        self._text_hashes = hashes
        self._reuse_stats = {"reused": len(hit_pos), "encoded": len(miss_texts)}
        self._index = TurboQuantIndex(dim=dim, bit_width=self._bit_width)
        self._index.add(out)

    def _ensure_embedder(self):
        # A from_dict-restored retriever has an index but no embedder yet;
        # the query encoder is only needed (and loaded) at search time.
        if self._embedder is None:
            SentenceTransformer, _TurboQuantIndex = self._require_backends()
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    def search_detailed(self, query: str, top_k: int) -> List[SearchHit]:
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
        return _hits_from_units(self._units, list(handles[0]))

    def search(self, query: str, top_k: int) -> List[str]:
        return [h.docid for h in self.search_detailed(query, top_k)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (docids + base64 float32 vectors).

        The quantized ``TurboQuantIndex`` itself is not serializable, so the
        pre-quantization vectors are stored and the index is rebuilt from
        them in ``from_dict``. ``schema`` stays the first key (the cheap
        head probe in ``retrieval.persistence`` relies on it).
        """
        if self._index is None or self._vectors is None:
            raise RuntimeError("call index() before to_dict()")
        data: Dict[str, Any] = {
            "schema": self.SCHEMA_VERSION,
            "model_name": self._model_name,
            "bit_width": self._bit_width,
            "docids": self._docids,
            "units": self._units,
        }
        if self._text_hashes is not None:
            data["text_sha256"] = self._text_hashes
        data["vectors"] = self._encode_vectors(self._vectors)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurbovecRetriever":
        """Rebuild from ``to_dict`` output; needs the same extras as ``index``.

        Accepts schema 3 and the legacy schema 2 (float lists; no hashes, so
        no embedding reuse until the next rebuild). Raises ``ValueError`` on
        an unrecognized schema or malformed vectors and ``RuntimeError``
        (with opt-in guidance) when the optional backends are missing.
        """
        schema = data.get("schema")
        if schema not in cls.LOADABLE_SCHEMAS:
            raise ValueError(f"unsupported TurbovecRetriever schema {schema!r}")
        retriever = cls(model_name=data["model_name"], bit_width=data["bit_width"])
        _SentenceTransformer, TurboQuantIndex = retriever._require_backends()

        vectors = cls._decode_vectors(data["vectors"])
        docids = data["docids"]
        if vectors.shape[0] != len(docids):
            raise ValueError("turbovec vector rows do not match docids")
        hashes = data.get("text_sha256") if schema >= 3 else None
        if hashes is not None and len(hashes) != len(docids):
            raise ValueError("turbovec text_sha256 length does not match docids")
        retriever._docids = docids
        retriever._units = data["units"]
        retriever._vectors = vectors
        retriever._text_hashes = hashes
        retriever._index = TurboQuantIndex(dim=vectors.shape[1], bit_width=retriever._bit_width)
        retriever._index.add(vectors)
        return retriever


class PiSeriniRetriever:
    """Lucene BM25 via Pyserini — the paper's reference lexical retriever.

    Builds an in-memory Lucene index over the corpus and queries it with
    Pyserini's ``LuceneSearcher``.  Requires Java 21 (Pyserini wraps Anserini).

    ``pi-serini`` (the strategy and registry key, from the Pi-Serini paper)
    and ``pyserini`` (the library and install extra) are distinct names,
    not a typo for each other.
    """

    name = "pi-serini (lucene bm25)"

    #: Hyperparameter keys this retriever accepts, as exposed to the CLI/
    #: autotune/``relevant_params`` — named ``lucene_*`` (rather than the
    #: constructor's own ``k1``/``b``) to disambiguate from the lexical
    #: arm's ``bm25_k1``/``bm25_b`` when both are surfaced together (e.g. a
    #: consolidated ``tune``/``stats`` report); mapped back to ``k1``/``b``
    #: in ``build_retriever`` via ``_CTOR_KWARG_MAP``. The constructor's own
    #: keyword names stay ``k1``/``b`` for backward compatibility (see
    #: ``from_dict``, which still does ``cls(k1=data["k1"], ...)``).
    ACCEPTS = frozenset({"lucene_k1", "lucene_b"})

    #: Bump when the persisted dict shape changes incompatibly (see
    #: ``LexicalRetriever.SCHEMA_VERSION``). v2 adds ``units`` (chunk span
    #: metadata).
    SCHEMA_VERSION = 2

    #: See ``LexicalRetriever.REQUIRED_MODULES``.
    REQUIRED_MODULES: tuple = ("pyserini",)

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
        self._units: List[Dict[str, Any]] = []
        self._units_by_docid: Dict[str, Dict[str, Any]] = {}

    @property
    def doc_count(self) -> int:
        """Number of indexed chunks (== ``len(to_dict()["docids"])``)."""
        return len(self._docids)

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
        # Lucene doc id is the chunk docid ("path:start-end"); spans are
        # resolved via self._units_by_docid at search time, never by parsing
        # the id string back apart.
        indexer.add_batch_dict(
            [{"id": d.docid, "contents": d.text} for d in documents]
        )
        indexer.close()
        self._units = _units_from_documents(documents)
        self._docids = [u["docid"] for u in self._units]
        self._units_by_docid = {u["docid"]: u for u in self._units}
        self._searcher = LuceneSearcher(self._index_dir)
        self._searcher.set_bm25(self._k1, self._b)

    def search_detailed(self, query: str, top_k: int) -> List[SearchHit]:
        if self._searcher is None:
            raise RuntimeError("call index() before search()")
        hits = self._searcher.search(query, k=top_k)
        results = []
        for rank, hit in enumerate(hits):
            unit = self._units_by_docid[hit.docid]
            results.append(
                SearchHit(
                    docid=unit["docid"],
                    source_path=unit["source_path"],
                    start_line=unit["start_line"],
                    end_line=unit["end_line"],
                    rank=rank,
                    metadata=dict(unit.get("metadata") or {}),
                )
            )
        return results

    def search(self, query: str, top_k: int) -> List[str]:
        return [h.docid for h in self.search_detailed(query, top_k)]

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
            "units": self._units,
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
        retriever._units = data["units"]
        retriever._units_by_docid = {u["docid"]: u for u in retriever._units}
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

    #: Hyperparameter keys this retriever accepts: the union of both arms'
    #: keys plus ``hybrid_weights`` (an optional ``[lexical_weight,
    #: dense_weight]`` pair overriding the arms' equal RRF weighting at
    #: search time). Split back out to each arm's own kwargs in ``__init__``.
    ACCEPTS = LexicalRetriever.ACCEPTS | TurbovecRetriever.ACCEPTS | frozenset({"hybrid_weights"})

    #: Bump when the persisted dict shape changes incompatibly (see
    #: ``LexicalRetriever.SCHEMA_VERSION``). v2 adds ``units`` (chunk span
    #: metadata). v3 tracks the lexical arm's own v3 bump (tokenizer mode).
    #: v4 nests the dense arm's v3 payload (base64 vectors + text hashes).
    SCHEMA_VERSION = 4

    #: Schemas ``from_dict`` accepts; v3 nests a dense v2 arm, which still loads.
    LOADABLE_SCHEMAS = frozenset({3, 4})

    #: See ``LexicalRetriever.REQUIRED_MODULES``.
    REQUIRED_MODULES: tuple = ("sentence_transformers", "turbovec", "numpy")

    def __init__(self, dense: Optional[TurbovecRetriever] = None, **params: Any) -> None:
        hybrid_weights = params.pop("hybrid_weights", None)
        lexical_params = {k: v for k, v in params.items() if k in LexicalRetriever.ACCEPTS}
        dense_params = {k: v for k, v in params.items() if k in TurbovecRetriever.ACCEPTS}
        self._lexical = LexicalRetriever(**lexical_params)
        self._dense = dense if dense is not None else TurbovecRetriever(**dense_params)
        self._hybrid_weights: Optional[List[float]] = (
            list(hybrid_weights) if hybrid_weights is not None else None
        )
        self._units_by_docid: Dict[str, Dict[str, Any]] = {}

    @property
    def doc_count(self) -> int:
        """Number of indexed chunks (== ``len(to_dict()["docids"])``)."""
        return len(self._lexical._docids)

    def set_embedding_seed(
        self, provider: Optional[Callable[[], _SeedResult]]
    ) -> None:
        """Forward the embedding-reuse seed to the dense arm (if it has one)."""
        setter = getattr(self._dense, "set_embedding_seed", None)
        if setter is not None:
            setter(provider)

    def arms_compatible(
        self,
        *,
        lexical: Optional[LexicalRetriever] = None,
        dense: Optional[Any] = None,
    ) -> bool:
        """True if pre-built *lexical*/*dense* arms equal what a fresh
        ``HybridRetriever`` with this one's constructor params would build.

        Exact ``LexicalRetriever`` only (subclasses such as the contextual or
        tree-sitter variants index different text).
        """
        if lexical is not None and not (
            type(lexical) is LexicalRetriever and lexical._params == self._lexical._params
        ):
            return False
        if dense is not None and not (
            getattr(dense, "_model_name", _MISSING) == self._dense._model_name
            and getattr(dense, "_bit_width", _MISSING) == self._dense._bit_width
        ):
            return False
        return True

    def adopt_arms(
        self,
        documents: List[Document],
        *,
        dense: Any,
        lexical: Optional[LexicalRetriever] = None,
    ) -> None:
        """Equivalent to ``index(documents)`` using arms already indexed over
        exactly *documents* (no re-embedding). A missing *lexical* arm is
        indexed fresh (cheap). Raises ``ValueError`` if the arms' params
        don't match this retriever's.
        """
        if not self.arms_compatible(lexical=lexical, dense=dense):
            raise ValueError("pre-built arms do not match this hybrid retriever's params")
        self._dense = dense
        if lexical is not None:
            self._lexical = lexical
        else:
            self._lexical.index(documents)
        self._units_by_docid = {u["docid"]: u for u in _units_from_documents(documents)}

    def index(self, documents: List[Document]) -> None:
        # Dense arm first: it fails fast (with opt-in guidance) when the
        # turbovec extras are missing, before any lexical work is done.
        # Both arms index the identical chunk-Documents, so RRF-by-docid
        # fusion below is unaffected by span metadata.
        self._dense.index(documents)
        self._lexical.index(documents)
        units = _units_from_documents(documents)
        self._units_by_docid = {u["docid"]: u for u in units}

    def search_detailed(self, query: str, top_k: int) -> List[SearchHit]:
        # Pull a deeper candidate pool from each arm than the caller asked
        # for, so RRF has overlap to work with before truncating to top_k.
        # Capped at this retriever's own indexed unit count (never deeper
        # than the corpus itself).
        pool = candidate_pool(top_k, n_units=len(self._units_by_docid))
        lexical_docids = self._lexical.search(query, pool)
        dense_docids = self._dense.search(query, pool)
        all_docids = sorted(set(lexical_docids) | set(dense_docids))
        to_idx = {docid: i for i, docid in enumerate(all_docids)}
        fused = reciprocal_rank_fusion(
            [
                [to_idx[d] for d in lexical_docids],
                [to_idx[d] for d in dense_docids],
            ],
            weights=self._hybrid_weights,
        )
        results = []
        for rank, (idx, _score) in enumerate(fused[:top_k]):
            docid = all_docids[idx]
            unit = self._units_by_docid[docid]
            results.append(
                SearchHit(
                    docid=unit["docid"],
                    source_path=unit["source_path"],
                    start_line=unit["start_line"],
                    end_line=unit["end_line"],
                    rank=rank,
                    metadata=dict(unit.get("metadata") or {}),
                )
            )
        return results

    def search(self, query: str, top_k: int) -> List[str]:
        return [h.docid for h in self.search_detailed(query, top_k)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize both arms to a JSON-safe dict for on-disk persistence."""
        lexical_data = self._lexical.to_dict()
        data = {
            "schema": self.SCHEMA_VERSION,
            "docids": lexical_data["docids"],
            "units": lexical_data["units"],
            "lexical": lexical_data,
            "dense": self._dense.to_dict(),
        }
        if self._hybrid_weights is not None:
            data["hybrid_weights"] = self._hybrid_weights
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridRetriever":
        """Rebuild both arms from ``to_dict`` output.

        Raises ``ValueError`` on an unrecognized schema version and
        ``RuntimeError`` when the dense arm's extras are missing.
        """
        schema = data.get("schema")
        if schema not in cls.LOADABLE_SCHEMAS:
            raise ValueError(f"unsupported HybridRetriever schema {schema!r}")
        retriever = cls()
        retriever._lexical = LexicalRetriever.from_dict(data["lexical"])
        retriever._dense = TurbovecRetriever.from_dict(data["dense"])
        retriever._units_by_docid = {u["docid"]: u for u in data["units"]}
        retriever._hybrid_weights = data.get("hybrid_weights")
        return retriever


# Registry keyed by CLI name.
REGISTRY = {
    "lexical": LexicalRetriever,
    "lexical+ctx": ContextualLexicalRetriever,
    "turbovec": TurbovecRetriever,
    "pi-serini": PiSeriniRetriever,
    "hybrid": HybridRetriever,
    "treesitter": TreeSitterRetriever,
}


#: Retriever name -> {public param key: constructor keyword} overrides,
#: where a retriever's CLI/autotune-facing param name differs from its
#: actual constructor keyword (kept stable for backward compatibility —
#: see ``PiSeriniRetriever.ACCEPTS``).
_CTOR_KWARG_MAP: Dict[str, Dict[str, str]] = {
    "pi-serini": {"lucene_k1": "k1", "lucene_b": "b"},
}


def resolve_ctor_kwargs(name: str, cls: type, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter *params* down to *cls*'s ``ACCEPTS`` (empty set if undeclared)
    and map each key to its constructor keyword name (see ``_CTOR_KWARG_MAP``).

    Unknown keys are silently dropped. ``params=None`` (or empty) yields an
    empty kwargs dict, reproducing a bare ``cls()`` construction.
    """
    if not params:
        return {}
    accepts = getattr(cls, "ACCEPTS", frozenset())
    mapping = _CTOR_KWARG_MAP.get(name, {})
    return {mapping.get(k, k): v for k, v in params.items() if k in accepts}


def build_retriever(name: str, params: Optional[Dict[str, Any]] = None) -> Retriever:
    """Construct retriever *name*, optionally applying *params*.

    *params* is filtered to the retriever class's own ``ACCEPTS`` and mapped
    to constructor keywords (see ``resolve_ctor_kwargs``); unknown keys are
    silently dropped. ``params=None`` (the default) reproduces a bare,
    default-hyperparameter construction, identical to before this parameter
    existed.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown retriever {name!r}; choose from {list(REGISTRY)}")
    cls = REGISTRY[name]
    return cls(**resolve_ctor_kwargs(name, cls, params))
