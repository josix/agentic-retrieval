"""Tests for retrieval.persistence.

Covers: to_dict/from_dict fidelity (BM25Index, TfidfIndex, LexicalRetriever)
including round-trip search equality; fingerprint stability and sensitivity
to content/add/remove changes; is_stale before/after a file touch;
load_index's None-on-missing/corrupt behavior; project_key uniqueness per
root; and the RETRIEVAL_INDEX_DIR env override.
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.bm25 import BM25Index  # noqa: E402
from retrieval.document import Document  # noqa: E402
from retrieval.persistence import (  # noqa: E402
    _loader_kw_from_meta,
    cache_base_dir,
    cached_retrievers,
    compute_fingerprint,
    index_dir,
    is_stale,
    load_index,
    project_key,
    relevant_params,
    save_index,
)
from retrieval.project_loader import (  # noqa: E402
    discover_files,
    load_chunk_documents,
    load_documents,
)
from retrieval.retrievers import (  # noqa: E402
    HybridRetriever,
    LexicalRetriever,
    TreeSitterRetriever,
    TurbovecRetriever,
)
from retrieval.tfidf import TfidfIndex  # noqa: E402

try:
    import turbovec  # noqa: F401

    _TURBOVEC_INSTALLED = True
except ImportError:
    _TURBOVEC_INSTALLED = False

try:
    import pyserini  # noqa: F401

    _PYSERINI_INSTALLED = True
except ImportError:
    _PYSERINI_INSTALLED = False


def _write(path: pathlib.Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bump_mtime(path: pathlib.Path) -> None:
    """Advance *path*'s mtime by 1s (in ns) so a fingerprint change is guaranteed
    regardless of filesystem mtime resolution."""
    new_ns = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(new_ns, new_ns))


class TestPersistence(unittest.TestCase):
    """Shared read-only project fixture; RETRIEVAL_INDEX_DIR is redirected per test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.tmpdir.name) / "project"
        _write(cls.root / "a.txt", "routers forward packets between networks and carry data")
        _write(cls.root / "b.txt", "photosynthesis converts sunlight into chemical energy")
        _write(cls.root / "sub" / "c.md", "the asteroid belt lies between mars and jupiter")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def setUp(self) -> None:
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        os.environ["RETRIEVAL_INDEX_DIR"] = self._cache_tmp.name

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        else:
            os.environ["RETRIEVAL_INDEX_DIR"] = self._old_env
        self._cache_tmp.cleanup()

    # -- to_dict/from_dict fidelity ---------------------------------------

    def test_tfidf_to_dict_from_dict_fidelity(self) -> None:
        docs = [d.text for d in load_documents(self.root)]
        index = TfidfIndex()
        index.fit(docs)
        data = index.to_dict()
        json.dumps(data)  # must be JSON-safe
        restored = TfidfIndex.from_dict(data)
        for query in ("routers packets", "photosynthesis energy", "asteroid belt"):
            self.assertEqual(index.query(query), restored.query(query))

    def test_bm25_to_dict_from_dict_fidelity(self) -> None:
        docs = [d.text for d in load_documents(self.root)]
        index = BM25Index(k1=1.3, b=0.8)
        index.fit(docs)
        data = index.to_dict()
        json.dumps(data)  # must be JSON-safe
        restored = BM25Index.from_dict(data)
        self.assertEqual(restored.k1, 1.3)
        self.assertEqual(restored.b, 0.8)
        for query in ("routers packets", "photosynthesis energy", "asteroid belt"):
            self.assertEqual(index.query(query), restored.query(query))

    def test_lexical_retriever_round_trip_search_equality(self) -> None:
        docs = load_documents(self.root)
        retriever = LexicalRetriever()
        retriever.index(docs)
        data = retriever.to_dict()
        json.dumps(data)  # must be JSON-safe
        restored = LexicalRetriever.from_dict(data)
        for query in (
            "what carries data between networks",
            "how do plants convert light into energy",
            "asteroid belt mars jupiter",
            "no shared vocabulary whatsoever xyzzy",
        ):
            self.assertEqual(retriever.search(query, top_k=3), restored.search(query, top_k=3))

    def test_lexical_retriever_from_dict_rejects_unknown_schema(self) -> None:
        docs = load_documents(self.root)
        retriever = LexicalRetriever()
        retriever.index(docs)
        data = retriever.to_dict()
        data["schema"] = 999
        with self.assertRaises(ValueError):
            LexicalRetriever.from_dict(data)

    def test_stale_v1_schema_cache_forces_rebuild(self) -> None:
        """A v1 (pre-units) on-disk cache must not be mis-parsed: load_index
        treats an unrecognized schema as "no usable cache", so callers fall
        back to a fresh rebuild rather than crashing or silently missing spans."""
        docs = load_documents(self.root)
        retriever = LexicalRetriever()
        retriever.index(docs)
        save_index(retriever, self.root, compute_fingerprint(self.root), "lexical", "0.2.0")

        directory = index_dir(self.root)
        data = json.loads((directory / "lexical.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], 3)
        # Simulate a stale v1 cache (no "units" key, old schema number).
        data["schema"] = 1
        del data["units"]
        (directory / "lexical.json").write_text(json.dumps(data), encoding="utf-8")

        self.assertIsNone(load_index(self.root, "lexical"))

    def test_genuine_v2_schema_payload_forces_rebuild(self) -> None:
        """A real v2 (pre-tokenizer-mode) on-disk cache — same shape as
        today's v3 except for the schema number itself — must also be
        rejected, not silently mis-parsed as v3."""
        docs = load_documents(self.root)
        retriever = LexicalRetriever()
        retriever.index(docs)
        save_index(retriever, self.root, compute_fingerprint(self.root), "lexical", "0.5.0")

        directory = index_dir(self.root)
        data = json.loads((directory / "lexical.json").read_text(encoding="utf-8"))
        data["schema"] = 2
        (directory / "lexical.json").write_text(json.dumps(data), encoding="utf-8")

        self.assertIsNone(load_index(self.root, "lexical"))

    # -- fingerprint --------------------------------------------------------

    def test_fingerprint_is_stable_across_calls(self) -> None:
        self.assertEqual(compute_fingerprint(self.root), compute_fingerprint(self.root))

    def test_fingerprint_changes_on_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            path = root / "x.txt"
            _write(path, "hello")
            fp1 = compute_fingerprint(root)
            _write(path, "hello world, much longer now")
            _bump_mtime(path)
            fp2 = compute_fingerprint(root)
            self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_on_file_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello")
            fp1 = compute_fingerprint(root)
            _write(root / "y.txt", "world")
            fp2 = compute_fingerprint(root)
            self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_on_file_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello")
            _write(root / "y.txt", "world")
            fp1 = compute_fingerprint(root)
            (root / "y.txt").unlink()
            fp2 = compute_fingerprint(root)
            self.assertNotEqual(fp1, fp2)

    # -- is_stale -------------------------------------------------------------

    def test_is_stale_false_after_save_true_after_touch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            path = root / "x.txt"
            _write(path, "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(retriever, root, fingerprint, "lexical", "0.2.0")

            loaded = load_index(root)
            self.assertIsNotNone(loaded)
            _retriever, meta = loaded
            self.assertFalse(is_stale(root, meta))

            _bump_mtime(path)
            self.assertTrue(is_stale(root, meta))

    # -- hyperparams meta + is_stale(params) -----------------------------------

    def test_save_index_meta_gains_blocks_only_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(
                retriever, root, fingerprint, "lexical", "0.6.0",
                params={"code_chars": 400, "unrelated": 1},
                corpus_stats={"mean_chunk_tokens": 42},
                loader_kw={"extensions": frozenset({".py", ".md"})},
            )
            _retriever, meta = load_index(root)
            self.assertEqual(meta["hyperparams"], {"code_chars": 400})
            self.assertEqual(meta["corpus_stats"], {"mean_chunk_tokens": 42})
            self.assertEqual(sorted(meta["loader_kw"]["extensions"]), [".md", ".py"])

    def test_save_index_omits_blocks_when_not_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(retriever, root, fingerprint, "lexical", "0.6.0")
            _retriever, meta = load_index(root)
            self.assertNotIn("hyperparams", meta)
            self.assertNotIn("corpus_stats", meta)
            self.assertNotIn("loader_kw", meta)

    def test_is_stale_true_on_changed_relevant_param_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(
                retriever, root, fingerprint, "lexical", "0.6.0",
                params={"code_chars": 400},
            )
            _retriever, meta = load_index(root)
            self.assertFalse(is_stale(root, meta, params={"code_chars": 400}))
            self.assertTrue(is_stale(root, meta, params={"code_chars": 1200}))

    def test_is_stale_params_none_ignores_hyperparams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(
                retriever, root, fingerprint, "lexical", "0.6.0",
                params={"code_chars": 400},
            )
            _retriever, meta = load_index(root)
            # A changed hyperparam is invisible when params=None (default):
            # fingerprint-only behavior, unchanged from before this feature.
            self.assertFalse(is_stale(root, meta))

    def test_is_stale_pre_upgrade_meta_stale_under_any_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(retriever, root, fingerprint, "lexical", "0.6.0")  # no params
            _retriever, meta = load_index(root)
            self.assertNotIn("hyperparams", meta)
            self.assertFalse(is_stale(root, meta))  # params=None: fingerprint-only
            self.assertTrue(is_stale(root, meta, params={"code_chars": 400}))

    def test_relevant_params_excludes_bm25_k1_for_turbovec(self) -> None:
        self.assertEqual(
            relevant_params("turbovec", {"bm25_k1": 1.2, "model_name": "x"}),
            {"model_name": "x"},
        )

    def test_is_stale_true_on_changed_bm25_k1_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello world")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever(bm25_k1=1.2)
            retriever.index(load_documents(root))
            save_index(
                retriever, root, fingerprint, "lexical", "0.6.0",
                params={"bm25_k1": 1.2},
            )
            _retriever, meta = load_index(root)
            self.assertEqual(meta["hyperparams"], {"bm25_k1": 1.2})
            self.assertFalse(is_stale(root, meta, params={"bm25_k1": 1.2}))
            self.assertTrue(is_stale(root, meta, params={"bm25_k1": 1.8}))

    def test_relevant_params_includes_chunking_keys_for_every_retriever(self) -> None:
        for name in ("lexical", "turbovec", "pi-serini", "hybrid", "treesitter"):
            with self.subTest(name=name):
                self.assertEqual(
                    relevant_params(name, {"code_chars": 1200, "unrelated_key": 1}),
                    {"code_chars": 1200},
                )

    def test_loader_kw_from_meta_rehydrates_frozensets(self) -> None:
        meta = {
            "loader_kw": {
                "extensions": [".py", ".md"],
                "exclude_dirs": [".git"],
                "include_basenames": ["readme"],
                "max_bytes": 500,
            }
        }
        result = _loader_kw_from_meta(meta)
        self.assertEqual(result["extensions"], frozenset({".py", ".md"}))
        self.assertEqual(result["exclude_dirs"], frozenset({".git"}))
        self.assertEqual(result["include_basenames"], frozenset({"readme"}))
        self.assertEqual(result["max_bytes"], 500)  # not a frozenset key: passed through

    def test_loader_kw_from_meta_empty_when_no_loader_kw_block(self) -> None:
        self.assertEqual(_loader_kw_from_meta({}), {})

    # -- load_index -----------------------------------------------------------

    def test_load_index_none_on_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "never-indexed"
            root.mkdir()
            self.assertIsNone(load_index(root))

    def test_load_index_none_on_corrupt_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "x.txt", "hello")
            fingerprint = compute_fingerprint(root)
            retriever = LexicalRetriever()
            retriever.index(load_documents(root))
            save_index(retriever, root, fingerprint, "lexical", "0.2.0")

            directory = index_dir(root)
            (directory / "lexical.json").write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(load_index(root))

    # -- per-retriever cache slots ---------------------------------------------

    def test_load_index_none_for_uncached_retriever_name(self) -> None:
        retriever = LexicalRetriever()
        retriever.index(load_documents(self.root))
        save_index(retriever, self.root, compute_fingerprint(self.root), "lexical", "0.2.0")
        # A lexical cache must not satisfy a request for a different retriever.
        self.assertIsNone(load_index(self.root, "turbovec"))
        self.assertIsNone(load_index(self.root, "hybrid"))
        self.assertIsNone(load_index(self.root, "pi-serini"))

    def test_save_and_load_reject_unknown_retriever_name(self) -> None:
        retriever = LexicalRetriever()
        retriever.index(load_documents(self.root))
        with self.assertRaises(ValueError):
            save_index(retriever, self.root, "fp", "not-a-retriever", "0.2.0")
        with self.assertRaises(ValueError):
            load_index(self.root, "not-a-retriever")

    def test_cached_retrievers_lists_saved_slots(self) -> None:
        self.assertEqual(cached_retrievers(self.root), {})
        retriever = LexicalRetriever()
        retriever.index(load_documents(self.root))
        save_index(retriever, self.root, compute_fingerprint(self.root), "lexical", "0.2.0")
        cached = cached_retrievers(self.root)
        self.assertEqual(list(cached), ["lexical"])
        self.assertEqual(cached["lexical"]["doc_count"], 3)

    def test_doc_count_is_chunks_and_file_count_is_distinct_files(self) -> None:
        """doc_count counts chunk-Documents (one per span); file_count
        counts distinct source files those chunks came from."""
        chunk_docs = load_chunk_documents(self.root)
        retriever = LexicalRetriever()
        retriever.index(chunk_docs)
        save_index(retriever, self.root, compute_fingerprint(self.root), "lexical", "0.2.0")

        directory = index_dir(self.root)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["doc_count"], len(chunk_docs))
        self.assertEqual(meta["file_count"], 3)  # a.txt, b.txt, sub/c.md
        self.assertGreaterEqual(meta["doc_count"], meta["file_count"])

    def test_lexical_ctx_shares_the_lexical_slot(self) -> None:
        retriever = LexicalRetriever()
        retriever.index(load_documents(self.root))
        save_index(retriever, self.root, compute_fingerprint(self.root), "lexical+ctx", "0.2.0")
        # Same files as lexical, so the slot reports the ctx name and a
        # plain lexical load still round-trips it.
        self.assertEqual(list(cached_retrievers(self.root)), ["lexical+ctx"])
        self.assertIsNotNone(load_index(self.root, "lexical"))

    def test_treesitter_round_trip_search_equality_and_context(self) -> None:
        """TreeSitterRetriever needs no optional extras (only the ast_chunker
        loader does), so this exercises the cache round-trip with inline
        Documents carrying a context breadcrumb."""
        docs = [
            Document(
                "d1", "def baz(self):\n    return self.value\n",
                source_path="pkg/bar.py", start_line=10, end_line=11,
                context="Bar.baz",
            ),
            Document(
                "d2", "def qux():\n    return 42\n",
                source_path="pkg/other.py", start_line=1, end_line=2,
            ),
        ]
        retriever = TreeSitterRetriever()
        retriever.index(docs)
        data = retriever.to_dict()
        json.dumps(data)  # must be JSON-safe
        self.assertEqual(data["units"][0]["context"], "Bar.baz")
        self.assertEqual(data["units"][1]["context"], "")

        save_index(retriever, self.root, compute_fingerprint(self.root), "treesitter", "0.2.0")
        loaded = load_index(self.root, "treesitter")
        self.assertIsNotNone(loaded)
        restored, _meta = loaded
        self.assertIsInstance(restored, TreeSitterRetriever)
        query = "baz self value"
        self.assertEqual(retriever.search(query, top_k=2), restored.search(query, top_k=2))
        hit = restored.search_detailed(query, top_k=1)[0]
        self.assertEqual(hit.context, "Bar.baz")

    @unittest.skipUnless(_TURBOVEC_INSTALLED, "turbovec not installed")
    def test_turbovec_round_trip_search_equality(self) -> None:
        docs = load_documents(self.root)
        retriever = TurbovecRetriever()
        retriever.index(docs)
        save_index(retriever, self.root, compute_fingerprint(self.root), "turbovec", "0.2.0")
        loaded = load_index(self.root, "turbovec")
        self.assertIsNotNone(loaded)
        restored, _meta = loaded
        query = "what carries data between networks"
        self.assertEqual(retriever.search(query, top_k=3), restored.search(query, top_k=3))

    @unittest.skipUnless(_TURBOVEC_INSTALLED, "turbovec not installed")
    def test_hybrid_round_trip_search_equality(self) -> None:
        docs = load_documents(self.root)
        retriever = HybridRetriever()
        retriever.index(docs)
        save_index(retriever, self.root, compute_fingerprint(self.root), "hybrid", "0.2.0")
        loaded = load_index(self.root, "hybrid")
        self.assertIsNotNone(loaded)
        restored, _meta = loaded
        query = "what carries data between networks"
        self.assertEqual(retriever.search(query, top_k=3), restored.search(query, top_k=3))

    @unittest.skipUnless(_PYSERINI_INSTALLED, "pyserini not installed")
    def test_pi_serini_round_trip_search_equality(self) -> None:
        from retrieval.retrievers import PiSeriniRetriever

        docs = load_documents(self.root)
        retriever = PiSeriniRetriever(index_path=index_dir(self.root) / "lucene")
        retriever.index(docs)
        save_index(retriever, self.root, compute_fingerprint(self.root), "pi-serini", "0.2.0")
        loaded = load_index(self.root, "pi-serini")
        self.assertIsNotNone(loaded)
        restored, _meta = loaded
        query = "what carries data between networks"
        self.assertEqual(retriever.search(query, top_k=3), restored.search(query, top_k=3))

    # -- project_key / cache_base_dir ------------------------------------------

    def test_project_key_differs_per_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root_a = pathlib.Path(tmp) / "a"
            root_b = pathlib.Path(tmp) / "b"
            root_a.mkdir()
            root_b.mkdir()
            self.assertNotEqual(project_key(root_a), project_key(root_b))

    def test_retrieval_index_dir_override_honored(self) -> None:
        with tempfile.TemporaryDirectory() as override_dir:
            os.environ["RETRIEVAL_INDEX_DIR"] = override_dir
            self.assertEqual(cache_base_dir(), pathlib.Path(override_dir))
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                _write(root / "x.txt", "hello")
                directory = index_dir(root)
                self.assertTrue(str(directory).startswith(override_dir))

    def test_index_dir_defaults_to_in_project_root_without_override(self) -> None:
        old_env = os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp) / "project"
                root.mkdir()
                self.assertIsNone(cache_base_dir())
                self.assertEqual(index_dir(root), root.resolve() / ".agentic-retrieval")
        finally:
            if old_env is not None:
                os.environ["RETRIEVAL_INDEX_DIR"] = old_env

    def test_discover_files_excludes_default_cache_dir(self) -> None:
        """The in-root default cache dir must never feed back into discovery
        (avoids indexing the cache's own lexical.json/meta.json)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "a.txt", "hello world")
            _write(root / ".agentic-retrieval" / "lexical.json", "{}")
            _write(root / ".agentic-retrieval" / "meta.json", "{}")

            found = list(discover_files(root))
            self.assertTrue(any(p.name == "a.txt" for p in found))
            self.assertFalse(any(".agentic-retrieval" in p.parts for p in found))


if __name__ == "__main__":
    unittest.main()
