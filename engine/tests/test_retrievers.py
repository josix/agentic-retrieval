"""Offline tests for the document-level retriever adapters.

Uses a synthetic in-memory corpus so the lexical retrievers are exercised
without any dataset download.
"""

import hashlib
import json
import unittest
from unittest import mock

from retrieval.document import Document
from retrieval.retrievers import (
    REGISTRY,
    ContextualLexicalRetriever,
    HybridRetriever,
    LexicalRetriever,
    TreeSitterRetriever,
    TurbovecRetriever,
    build_retriever,
)

try:
    import numpy as np

    _NUMPY_INSTALLED = True
except ImportError:
    _NUMPY_INSTALLED = False

try:
    import turbovec  # noqa: F401

    _TURBOVEC_INSTALLED = True
except ImportError:
    _TURBOVEC_INSTALLED = False


def _documents():
    return [
        Document(
            "d1", "Routers forward packets between networks and carry data.",
            source_path="net.txt", start_line=1, end_line=1,
        ),
        Document(
            "d2", "Photosynthesis converts sunlight into chemical energy in plants.",
            source_path="bio.txt", start_line=2, end_line=2,
        ),
        Document(
            "d3", "The asteroid belt lies between Mars and Jupiter.",
            source_path="astro.txt", start_line=3, end_line=4,
        ),
        Document(
            "d4", "Switches direct frames within a local area network domain.",
            source_path="net.txt", start_line=5, end_line=5,
        ),
    ]


class TestLexicalRetriever(unittest.TestCase):
    def test_ranks_relevant_doc_first(self) -> None:
        documents = _documents()
        r = LexicalRetriever()
        r.index(documents)
        self.assertEqual(r.search("what carries data between networks", top_k=1)[0], "d1")
        self.assertEqual(r.search("how do plants convert light into energy", top_k=1)[0], "d2")

    def test_search_detailed_returns_populated_spans(self) -> None:
        documents = _documents()
        r = LexicalRetriever()
        r.index(documents)
        hits = r.search_detailed("what carries data between networks", top_k=1)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.docid, "d1")
        self.assertEqual(hit.source_path, "net.txt")
        self.assertEqual(hit.start_line, 1)
        self.assertEqual(hit.end_line, 1)
        self.assertEqual(hit.rank, 0)

    def test_search_is_consistent_with_search_detailed(self) -> None:
        documents = _documents()
        r = LexicalRetriever()
        r.index(documents)
        hits = r.search_detailed("asteroid belt mars jupiter", top_k=4)
        docids = r.search("asteroid belt mars jupiter", top_k=4)
        self.assertEqual([h.docid for h in hits], docids)


class TestContextualLexicalRetriever(unittest.TestCase):
    """LLM-contextualized lexical arm, exercised with a stub (no API)."""

    def test_registered_and_selectable(self) -> None:
        self.assertIn("lexical+ctx", REGISTRY)
        self.assertIs(REGISTRY["lexical+ctx"], ContextualLexicalRetriever)

    def test_enriched_text_is_searchable(self) -> None:
        documents = _documents()
        # Stub contextualizer injects a sentinel token derived from the docid;
        # no token appears in any raw document, so a hit proves enrichment ran.
        seen = []

        def stub(text: str) -> str:
            seen.append(text)
            return "ZQXENRICH"

        r = ContextualLexicalRetriever(contextualizer=stub)
        r.index(documents)
        self.assertEqual(len(seen), len(documents))  # one call per doc
        # Every doc now contains the sentinel, so it ranks for that token.
        self.assertEqual(len(r.search("ZQXENRICH", top_k=4)), 4)
        # Real query routing still works on the original text.
        self.assertEqual(r.search("what carries data between networks", top_k=1)[0], "d1")

    def test_enrichment_preserves_document_context_on_search_hit(self) -> None:
        # T-R1: a Document's `context` breadcrumb (e.g. a chunk heading, see
        # project_loader.load_chunk_documents) must survive enrichment and
        # come back on the SearchHit, same as source_path/start_line/end_line.
        documents = [
            Document(
                "d1", "Routers forward packets between networks and carry data.",
                source_path="net.txt", start_line=1, end_line=1, context="Networking",
            ),
        ]
        r = ContextualLexicalRetriever(contextualizer=lambda text: "CTX")
        r.index(documents)
        hit = r.search_detailed("what carries data between networks", top_k=1)[0]
        self.assertEqual(hit.context, "Networking")

    def test_enrichment_does_not_alter_span_metadata(self) -> None:
        """Enrichment only prepends context to `text`; span fields (and
        docid, source_path) must round-trip through index() unchanged."""
        documents = _documents()
        r = ContextualLexicalRetriever(contextualizer=lambda text: "CTX")
        r.index(documents)
        hit = r.search_detailed("what carries data between networks", top_k=1)[0]
        self.assertEqual(hit.docid, "d1")
        self.assertEqual(hit.source_path, "net.txt")
        self.assertEqual(hit.start_line, 1)
        self.assertEqual(hit.end_line, 1)


class TestTurbovecGracefulDegradation(unittest.TestCase):
    """When turbovec isn't installed, indexing must fail with a clear RuntimeError."""

    @unittest.skipIf(
        _TURBOVEC_INSTALLED,
        "turbovec is installed in this environment; degradation path not exercised",
    )
    def test_raises_runtime_error_when_uninstalled(self) -> None:
        r = build_retriever("turbovec")
        with self.assertRaises(RuntimeError):
            r.index(_documents())

    @unittest.skipUnless(
        _TURBOVEC_INSTALLED,
        "turbovec not installed; skipping installed-path assertion",
    )
    def test_registry_builds_a_turbovec_retriever_when_installed(self) -> None:
        r = build_retriever("turbovec")
        self.assertEqual(r.name, "turbovec (dense ann)")


class TestHybridRetriever(unittest.TestCase):
    """Lexical + dense RRF fusion; the dense arm is stubbed so no extras
    are needed — the real-turbovec path is covered only when installed."""

    def test_registered_and_selectable(self) -> None:
        self.assertIn("hybrid", REGISTRY)
        self.assertIs(REGISTRY["hybrid"], HybridRetriever)

    @unittest.skipIf(
        _TURBOVEC_INSTALLED,
        "turbovec is installed in this environment; degradation path not exercised",
    )
    def test_index_raises_runtime_error_when_turbovec_uninstalled(self) -> None:
        r = build_retriever("hybrid")
        with self.assertRaises(RuntimeError):
            r.index(_documents())

    def test_fuses_lexical_and_dense_rankings(self) -> None:
        documents = _documents()

        class StubDense:
            """Dense arm that always ranks d3 first, then d1."""

            def index(self, docs) -> None:
                self.indexed = [d.docid for d in docs]

            def search(self, query: str, top_k: int):
                return ["d3", "d1"][:top_k]

        stub = StubDense()
        r = HybridRetriever(dense=stub)
        r.index(documents)
        self.assertEqual(stub.indexed, [d.docid for d in documents])

        results = r.search("what carries data between networks", top_k=4)
        # d1 appears high in both arms, so fusion must rank it first; the
        # stub's d3 must also surface even though lexical ranks it low.
        self.assertEqual(results[0], "d1")
        self.assertIn("d3", results[:2])

    def test_search_detailed_fuses_by_chunk_docid_with_spans(self) -> None:
        documents = _documents()

        class StubDense:
            def index(self, docs) -> None:
                self.indexed = [d.docid for d in docs]

            def search(self, query: str, top_k: int):
                return ["d3", "d1"][:top_k]

        stub = StubDense()
        r = HybridRetriever(dense=stub)
        r.index(documents)

        hits = r.search_detailed("what carries data between networks", top_k=4)
        self.assertEqual(hits[0].docid, "d1")
        self.assertEqual(hits[0].source_path, "net.txt")
        self.assertEqual(hits[0].start_line, 1)
        self.assertEqual(hits[0].end_line, 1)
        self.assertEqual(hits[0].rank, 0)
        # search() must be the plain-docid projection of search_detailed().
        self.assertEqual(
            [h.docid for h in hits],
            r.search("what carries data between networks", top_k=4),
        )


class TestBuildRetrieverParams(unittest.TestCase):
    def test_build_retriever_filters_and_maps_params(self) -> None:
        r = build_retriever("lexical", {"bm25_k1": 1.2, "unknown": 9})
        self.assertIsInstance(r, LexicalRetriever)
        self.assertEqual(r._bm25.k1, 1.2)

    def test_build_retriever_none_params_reproduces_defaults(self) -> None:
        r = build_retriever("lexical")
        self.assertEqual(r._bm25.k1, 1.5)
        self.assertEqual(r._bm25.b, 0.75)
        self.assertEqual(r._tfidf.tokenizer, "plain")

    def test_tokenizer_round_trip_identical_ranking_pre_and_post_persist(self) -> None:
        documents = _documents()
        r = build_retriever("lexical", {"tokenizer": "code"})
        r.index(documents)
        data = r.to_dict()
        restored = LexicalRetriever.from_dict(data)
        self.assertEqual(restored._tfidf.tokenizer, "code")
        self.assertEqual(restored._bm25.tokenizer, "code")
        for query in ("what carries data between networks", "asteroid belt"):
            self.assertEqual(r.search(query, top_k=3), restored.search(query, top_k=3))

    def test_hybrid_forwards_params_to_correct_arms(self) -> None:
        r = build_retriever(
            "hybrid", {"bm25_k1": 1.2, "model_name": "some/model", "unknown": 9}
        )
        self.assertIsInstance(r, HybridRetriever)
        self.assertEqual(r._lexical._bm25.k1, 1.2)
        self.assertEqual(r._dense._model_name, "some/model")


class TestTreeSitterRetriever(unittest.TestCase):
    """LexicalRetriever over AST-chunk Documents; no tree-sitter needed here —
    span/context metadata is fed in directly via inline Documents."""

    def _code_documents(self):
        return [
            Document(
                "d1", "def baz(self):\n    return self.value * 2\n",
                source_path="pkg/bar.py", start_line=10, end_line=11,
                context="Bar.baz",
            ),
            Document(
                "d2", "def qux():\n    return 42\n",
                source_path="pkg/other.py", start_line=1, end_line=2,
                context="",
            ),
        ]

    def test_registered_and_selectable(self) -> None:
        self.assertIn("treesitter", REGISTRY)
        self.assertIs(REGISTRY["treesitter"], TreeSitterRetriever)
        self.assertIsInstance(build_retriever("treesitter"), TreeSitterRetriever)

    def test_ranks_relevant_chunk_first(self) -> None:
        documents = self._code_documents()
        r = TreeSitterRetriever()
        r.index(documents)
        self.assertEqual(r.search("baz self value", top_k=1)[0], "d1")

    def test_search_detailed_returns_spans_and_context(self) -> None:
        documents = self._code_documents()
        r = TreeSitterRetriever()
        r.index(documents)
        hits = r.search_detailed("baz self value", top_k=1)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.docid, "d1")
        self.assertEqual(hit.source_path, "pkg/bar.py")
        self.assertEqual(hit.start_line, 10)
        self.assertEqual(hit.end_line, 11)
        self.assertEqual(hit.context, "Bar.baz")

    def test_breadcrumb_is_searchable(self) -> None:
        """A query that only matches the breadcrumb (not the chunk body)
        must still surface the chunk, proving context was indexed."""
        documents = self._code_documents()
        r = TreeSitterRetriever()
        r.index(documents)
        self.assertEqual(r.search("Bar baz", top_k=1)[0], "d1")

    def test_empty_context_leaves_hit_context_blank(self) -> None:
        documents = self._code_documents()
        r = TreeSitterRetriever()
        r.index(documents)
        hits = r.search_detailed("qux 42", top_k=1)
        self.assertEqual(hits[0].docid, "d2")
        self.assertEqual(hits[0].context, "")


class TestRetrieverMetadata(unittest.TestCase):
    def _docs(self):
        return [
            Document("d1", "iterator types 疊代器", source_path="a.po", start_line=1, end_line=2,
                     metadata={"kind": "po_entry", "msgid": "Iterator Types"}),
            Document("d2", "logging root", source_path="b.po", start_line=1, end_line=2),
        ]

    def test_hits_carry_metadata(self) -> None:
        r = LexicalRetriever(tokenizer="cjk")
        r.index(self._docs())
        hit = r.search_detailed("疊代器", top_k=1)[0]
        self.assertEqual(hit.metadata["msgid"], "Iterator Types")
        plain = r.search_detailed("logging root", top_k=1)[0]
        self.assertEqual(plain.metadata, {})
        self.assertNotIn("metadata", r.to_dict()["units"][1])

    def test_roundtrip_and_old_cache_without_key(self) -> None:
        r = LexicalRetriever(tokenizer="cjk")
        r.index(self._docs())
        data = r.to_dict()
        restored = LexicalRetriever.from_dict(data)
        top = restored.search_detailed("疊代器", top_k=1)[0]
        self.assertEqual(top.metadata["kind"], "po_entry")
        for unit in data["units"]:
            unit.pop("metadata", None)
        old = LexicalRetriever.from_dict(data)
        self.assertEqual(old.search_detailed("疊代器", top_k=1)[0].metadata, {})

    def test_treesitter_and_contextual_preserve_metadata(self) -> None:
        contextual = ContextualLexicalRetriever(contextualizer=lambda t: "ctx")
        for retriever in (TreeSitterRetriever(), contextual):
            retriever.index(self._docs())
            hit = retriever.search_detailed("iterator types", top_k=1)[0]
            self.assertEqual(hit.metadata["msgid"], "Iterator Types")


class _StubDense:
    """Deterministic stand-in for TurbovecRetriever (no embedding model)."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", bit_width=4):
        self._model_name = model_name
        self._bit_width = bit_width
        self._docids = []
        self.index_calls = 0

    @property
    def doc_count(self):
        return len(self._docids)

    def index(self, documents):
        self.index_calls += 1
        self._docids = [d.docid for d in documents]

    def search(self, query, top_k):
        return sorted(self._docids, key=lambda d: (len(d) + len(query)) % 3)[:top_k]

    def to_dict(self):
        return {"schema": 2, "model_name": self._model_name, "bit_width": self._bit_width,
                "docids": self._docids, "units": [], "vectors": []}


class TestDocCount(unittest.TestCase):
    def test_doc_count_matches_to_dict_docids(self) -> None:
        for cls in (LexicalRetriever, TreeSitterRetriever):
            with self.subTest(cls=cls.__name__):
                r = cls()
                r.index(_documents())
                self.assertEqual(r.doc_count, len(r.to_dict()["docids"]))
                self.assertGreater(r.doc_count, 0)

    def test_hybrid_doc_count_matches_to_dict(self) -> None:
        r = HybridRetriever(dense=_StubDense())
        r.index(_documents())
        self.assertEqual(r.doc_count, len(r.to_dict()["docids"]))


class TestHybridAdoptArms(unittest.TestCase):
    def test_adopt_arms_matches_full_index(self) -> None:
        docs = _documents()
        full = HybridRetriever(dense=_StubDense())
        full.index(docs)

        lexical, dense = LexicalRetriever(), _StubDense()
        lexical.index(docs)
        dense.index(docs)
        adopted = HybridRetriever(dense=_StubDense())
        adopted.adopt_arms(docs, dense=dense, lexical=lexical)

        self.assertEqual(adopted.to_dict(), full.to_dict())
        for query in ("routers packets", "photosynthesis", "asteroid"):
            self.assertEqual(
                adopted.search_detailed(query, 3), full.search_detailed(query, 3)
            )
        self.assertEqual(dense.index_calls, 1)

    def test_adopt_arms_without_lexical_indexes_fresh_lexical_arm(self) -> None:
        docs = _documents()
        dense = _StubDense()
        dense.index(docs)
        adopted = HybridRetriever(dense=_StubDense())
        adopted.adopt_arms(docs, dense=dense)
        expected = LexicalRetriever()
        expected.index(docs)
        self.assertEqual(adopted.to_dict()["lexical"], expected.to_dict())

    def test_arms_compatible_rejects_mismatch(self) -> None:
        hybrid = HybridRetriever(dense=_StubDense())
        self.assertTrue(hybrid.arms_compatible(dense=_StubDense(), lexical=LexicalRetriever()))
        self.assertFalse(hybrid.arms_compatible(dense=_StubDense(bit_width=2)))
        self.assertFalse(hybrid.arms_compatible(dense=_StubDense(model_name="other")))
        self.assertFalse(hybrid.arms_compatible(lexical=LexicalRetriever(bm25_k1=1.2)))
        self.assertFalse(hybrid.arms_compatible(lexical=TreeSitterRetriever()))
        self.assertFalse(
            hybrid.arms_compatible(lexical=ContextualLexicalRetriever(contextualizer=str))
        )
        with self.assertRaises(ValueError):
            hybrid.adopt_arms(_documents(), dense=_StubDense(bit_width=2))


if __name__ == "__main__":
    unittest.main()


_DIM = 64


class _FakeST:
    """Deterministic embedder: unit vector seeded from sha256(text)."""

    encode_calls: list = []
    instances = 0

    def __init__(self, name):
        type(self).instances += 1

    def encode(self, texts, **_kw):
        type(self).encode_calls.append(list(texts))
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(_DIM)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows, dtype=np.float64).reshape(len(texts), _DIM)


class _FakeTQI:
    def __init__(self, dim, bit_width):
        self.dim = dim
        self.data = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors):
        self.data = np.vstack([self.data, vectors])

    def search(self, q, k):
        scores = q @ self.data.T
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, order, axis=1), order


def _chunk_docs(texts):
    return [
        Document(f"f.py:{i}-{i}", t, source_path="f.py", start_line=i, end_line=i)
        for i, t in enumerate(texts, 1)
    ]


@unittest.skipUnless(_NUMPY_INSTALLED, "numpy not installed")
class TestTurbovecIncremental(unittest.TestCase):
    def setUp(self) -> None:
        _FakeST.encode_calls = []
        _FakeST.instances = 0
        patcher = mock.patch.object(
            TurbovecRetriever, "_require_backends", return_value=(_FakeST, _FakeTQI)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build(self, texts, seed=None):
        r = TurbovecRetriever(model_name="m")
        if seed is not None:
            r.set_embedding_seed(lambda: seed)
        r.index(_chunk_docs(texts))
        return r

    def test_first_build_dedups_and_holds_ndarray(self) -> None:
        r = self._build(["a", "b", "a"])
        self.assertEqual(_FakeST.encode_calls, [["a", "b"]])
        self.assertIsInstance(r._vectors, np.ndarray)
        self.assertEqual(r._vectors.dtype, np.float32)
        self.assertEqual(r._vectors.shape, (3, _DIM))
        self.assertEqual(r._vectors[0].tobytes(), r._vectors[2].tobytes())

    def test_rebuild_encodes_only_changed_and_new(self) -> None:
        r1 = self._build(["one", "two", "three"])
        _FakeST.encode_calls = []
        r2 = self._build(["one", "TWO!", "three", "four"], seed=r1.to_dict())
        self.assertEqual(_FakeST.encode_calls, [["TWO!", "four"]])
        self.assertEqual(r2._reuse_stats, {"reused": 2, "encoded": 2})

    def test_reused_vectors_byte_identical_incl_json_roundtrip(self) -> None:
        r1 = self._build(["one", "two", "three"])
        seed = json.loads(json.dumps(r1.to_dict()))
        r2 = self._build(["three", "new", "one"], seed=seed)
        self.assertEqual(r2._vectors[0].tobytes(), r1._vectors[2].tobytes())
        self.assertEqual(r2._vectors[2].tobytes(), r1._vectors[0].tobytes())
        restored = TurbovecRetriever.from_dict(json.loads(json.dumps(r2.to_dict())))
        self.assertEqual(restored._vectors.tobytes(), r2._vectors.tobytes())

    def test_zero_misses_skips_embedder_until_search(self) -> None:
        r1 = self._build(["one", "two"])
        _FakeST.encode_calls = []
        _FakeST.instances = 0
        r2 = self._build(["two", "one"], seed=r1.to_dict())
        self.assertEqual(_FakeST.encode_calls, [])
        self.assertEqual(_FakeST.instances, 0)
        self.assertEqual(r2.search("one", 1), ["f.py:2-2"])
        self.assertEqual(_FakeST.instances, 1)

    def test_unusable_seeds_fall_back_to_full_encode(self) -> None:
        r1 = self._build(["one", "two"])
        good = r1.to_dict()
        v2 = {k: v for k, v in good.items() if k != "text_sha256"}
        v2["vectors"] = r1._vectors.tolist()
        seeds = [
            dict(good, model_name="other"),
            v2,
            dict(good, vectors=dict(good["vectors"], data="!!")),
            {"garbage": 1},
        ]
        for seed in seeds:
            _FakeST.encode_calls = []
            self._build(["one", "two"], seed=seed)
            self.assertEqual(_FakeST.encode_calls, [["one", "two"]])
        r = TurbovecRetriever(model_name="m")

        def boom():
            raise OSError("nope")

        r.set_embedding_seed(boom)
        r.index(_chunk_docs(["one"]))
        self.assertEqual(r._reuse_stats["encoded"], 1)

    def test_v3_round_trip_is_ndarray_not_list(self) -> None:
        r = self._build(["one", "two", "three"])
        data = json.loads(json.dumps(r.to_dict()))
        self.assertEqual(list(data)[0], "schema")
        self.assertIsInstance(data["vectors"], dict)
        restored = TurbovecRetriever.from_dict(data)
        v = restored._vectors
        self.assertNotIsInstance(v, list)
        self.assertEqual(v.dtype, np.float32)
        self.assertTrue(v.flags["C_CONTIGUOUS"])
        self.assertTrue(v.flags["WRITEABLE"])
        self.assertEqual(restored._text_hashes, r._text_hashes)

    def test_v3_malformed_payloads_raise_value_error(self) -> None:
        good = self._build(["one", "two"]).to_dict()
        bad_payloads = [
            dict(good, vectors=dict(good["vectors"], data="@@@@")),
            dict(good, vectors=dict(good["vectors"], data=None)),
            dict(good, vectors=dict(good["vectors"], shape=[2, _DIM + 1])),
            dict(good, vectors=dict(good["vectors"], shape=[3, _DIM])),
            dict(good, vectors=dict(good["vectors"], encoding="hex")),
            dict(good, text_sha256=["x"]),
        ]
        for bad in bad_payloads:
            with self.assertRaises(ValueError):
                TurbovecRetriever.from_dict(bad)

    def test_seed_iterable_skips_incompatible_candidates(self) -> None:
        other = TurbovecRetriever(model_name="other")
        other.index(_chunk_docs(["one", "two"]))
        mine = self._build(["one", "two"]).to_dict()
        _FakeST.encode_calls = []
        r = TurbovecRetriever(model_name="m")
        r.set_embedding_seed(lambda: iter([other.to_dict(), mine]))
        r.index(_chunk_docs(["one", "two", "new"]))
        self.assertEqual(_FakeST.encode_calls, [["new"]])

    def test_v2_cache_loads_without_reuse_and_saves_v3(self) -> None:
        r = self._build(["one", "two"])
        v2 = {
            "schema": 2, "model_name": "m", "bit_width": 4, "docids": r._docids,
            "units": r._units, "vectors": r._vectors.tolist(),
        }
        restored = TurbovecRetriever.from_dict(v2)
        self.assertIsInstance(restored._vectors, np.ndarray)
        self.assertIsNone(restored._text_hashes)
        out = restored.to_dict()
        self.assertEqual(out["schema"], 3)
        self.assertNotIn("text_sha256", out)
        with self.assertRaises(ValueError):
            TurbovecRetriever.from_dict(dict(v2, schema=1))


class TestHybridEmbeddingSeed(unittest.TestCase):
    def test_seed_forwarded_to_dense_and_stub_tolerated(self) -> None:
        dense = TurbovecRetriever()
        HybridRetriever(dense=dense).set_embedding_seed(lambda: None)
        self.assertIsNotNone(dense._seed_provider)
        HybridRetriever(dense=_StubDense()).set_embedding_seed(lambda: None)

    def test_hybrid_loadable_schemas(self) -> None:
        h = HybridRetriever(dense=_StubDense())
        h.index(_documents())
        data = h.to_dict()
        self.assertEqual(data["schema"], 4)
        for schema in (3, 4):
            restored_schema = dict(data, schema=schema)
            with mock.patch.object(
                TurbovecRetriever, "from_dict", return_value=_StubDense()
            ):
                HybridRetriever.from_dict(restored_schema)
        with self.assertRaises(ValueError):
            HybridRetriever.from_dict(dict(data, schema=2))
