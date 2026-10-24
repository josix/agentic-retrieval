"""In-process tests for retrieval.cli.main.

Exercises index/query/stats end-to-end against a real temp project tree,
redirecting RETRIEVAL_INDEX_DIR to a temp cache dir per test (restored in
tearDown) so nothing touches a developer's real ~/.cache. Output is
captured via contextlib.redirect_stdout rather than shelling out, since
main() is a plain in-process function.
"""

import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from test_extractors import _PYPDF_INSTALLED, _write_pdf  # noqa: E402
from test_record_loaders import register_fake_xyz  # noqa: E402
from test_retrievers import _StubDense  # noqa: E402

from retrieval import cli, extractors  # noqa: E402
from retrieval.cli import main  # noqa: E402
from retrieval.consolidation import ConsolidatedHit  # noqa: E402
from retrieval.document import Document, SearchHit  # noqa: E402
from retrieval.persistence import compute_fingerprint, index_dir, save_index  # noqa: E402
from retrieval.project_loader import load_chunk_documents  # noqa: E402
from retrieval.retrievers import LexicalRetriever  # noqa: E402

try:
    import turbovec  # noqa: F401

    _TURBOVEC_INSTALLED = True
except ImportError:
    _TURBOVEC_INSTALLED = False

try:
    import tree_sitter_language_pack  # noqa: F401

    _TREESITTER_INSTALLED = True
except ImportError:
    _TREESITTER_INSTALLED = False


def _write(path: pathlib.Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bump_mtime(path: pathlib.Path) -> None:
    """Advance *path*'s mtime by 1s (in ns) so a fingerprint change is guaranteed
    regardless of filesystem mtime resolution."""
    new_ns = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(new_ns, new_ns))


def _run(argv):
    """Run cli.main(argv), capturing stdout; returns (exit_code, stdout_text)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


def _run_with_stderr(argv):
    """Like _run, but also captures stderr; returns (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self._project_tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._project_tmp.name)
        _write(self.root / "a.txt", "routers forward packets between networks and carry data")
        _write(self.root / "b.txt", "photosynthesis converts sunlight into chemical energy")

        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        os.environ["RETRIEVAL_INDEX_DIR"] = self._cache_tmp.name

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        else:
            os.environ["RETRIEVAL_INDEX_DIR"] = self._old_env
        self._cache_tmp.cleanup()
        self._project_tmp.cleanup()

    def test_index_returns_zero_and_writes_cache_files(self) -> None:
        code, out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 chunks ->", out)
        self.assertIn("fingerprint=", out)

        directory = index_dir(self.root)
        self.assertTrue((directory / "lexical.json").exists())
        self.assertTrue((directory / "meta.json").exists())

    def test_index_without_force_is_a_noop_when_cache_is_fresh(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns

        # Second run, no --force, corpus unchanged: fast path, no rewrite.
        code, out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        self.assertIn("index up to date ->", out)
        self.assertIn("use --force to rebuild", out)
        self.assertEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

    def test_index_force_rebuilds_even_when_cache_is_fresh(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns

        time.sleep(0.01)  # ensure a detectable mtime/created_at difference
        code, out = _run(["index", "--root", str(self.root), "--retriever", "lexical", "--force"])
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 chunks ->", out)
        self.assertNotEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

    def test_index_without_force_rebuilds_when_corpus_is_stale(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns

        new_file = self.root / "c.txt"
        _write(new_file, "asteroid belt lies between mars and jupiter")
        _bump_mtime(new_file)

        code, out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        self.assertIn("indexed 3 chunks ->", out)
        self.assertNotEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

    def test_query_returns_zero_and_expected_docid(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt:1-1")

    def test_query_without_cache_auto_indexes(self) -> None:
        # No prior "index" call — query must build + persist the cache itself.
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt:1-1")
        directory = index_dir(self.root)
        self.assertTrue((directory / "lexical.json").exists())

    def test_query_stale_ok_without_cache_auto_indexes(self) -> None:
        # No cache at all, plus --stale-ok: staleness is irrelevant when
        # there's nothing to compare against, so this must still build +
        # persist the cache and return results, same as the no-flag case.
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--retriever", "lexical",
             "--top-k", "1", "--stale-ok"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt:1-1")
        directory = index_dir(self.root)
        self.assertTrue((directory / "lexical.json").exists())

    def test_stale_cache_auto_reindexes_with_new_content(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        # Add a new, more relevant file and bump its mtime so the fingerprint changes.
        new_file = self.root / "c.txt"
        _write(new_file, "asteroid belt lies between mars and jupiter")
        _bump_mtime(new_file)

        code, out = _run(
            ["query", "asteroid belt mars jupiter", "--root", str(self.root),
             "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "c.txt:1-1")

    def test_stale_ok_returns_stale_results_without_reindexing(self) -> None:
        code, _out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)

        new_file = self.root / "c.txt"
        _write(new_file, "asteroid belt lies between mars and jupiter")
        _bump_mtime(new_file)

        code, out = _run(
            [
                "query",
                "asteroid belt mars jupiter",
                "--root",
                str(self.root),
                "--top-k",
                "5",
                "--stale-ok",
            ]
        )
        self.assertEqual(code, 0)
        # Stale cache predates c.txt, so it can't possibly be returned.
        self.assertNotIn("c.txt", out.split())

    def test_stats_before_and_after_index(self) -> None:
        code, out = _run(["stats", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("no cache for", out)

        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        code, out = _run(["stats", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("chunks: 2", out)
        self.assertIn("files: 2", out)
        self.assertIn("stale: False", out)

    def test_json_output_parses(self) -> None:
        # Regression: an explicit `--retriever lexical` query must produce
        # byte-identical-shape JSON to the pre-consolidation-mode behavior.
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        code, out = _run(
            [
                "query",
                "what carries data between networks",
                "--root",
                str(self.root),
                "--retriever",
                "lexical",
                "--top-k",
                "1",
                "--json",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(set(payload.keys()), {"query", "results"})
        self.assertEqual(payload["query"], "what carries data between networks")
        self.assertEqual(len(payload["results"]), 1)
        result = payload["results"][0]
        self.assertEqual(
            set(result.keys()), {"docid", "path", "start_line", "end_line", "rank", "context"}
        )
        self.assertEqual(result["docid"], "a.txt:1-1")
        self.assertEqual(result["path"], "a.txt")
        self.assertEqual(result["start_line"], 1)
        self.assertEqual(result["end_line"], 1)
        self.assertEqual(result["rank"], 0)

    def test_consolidated_default_json_envelope(self) -> None:
        code, out = _run(
            [
                "query",
                "what carries data between networks",
                "--root",
                str(self.root),
                "--top-k",
                "1",
                "--json",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["query"], "what carries data between networks")
        self.assertEqual(payload["mode"], "consolidated")
        self.assertIn("lexical", payload["retrievers"])
        self.assertIsInstance(payload["skipped"], list)
        self.assertGreaterEqual(len(payload["results"]), 1)
        result = payload["results"][0]
        for key in (
            "docid", "path", "start_line", "end_line", "rank", "context",
            "score", "provenance", "agreement", "confidence", "contributors",
        ):
            self.assertIn(key, result)

    def test_consolidated_default_retriever_choice_is_explicit_alias(self) -> None:
        code_default, out_default = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--top-k", "1", "--json"]
        )
        code_all, out_all = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--retriever", "all", "--top-k", "1", "--json"]
        )
        self.assertEqual(code_default, 0)
        self.assertEqual(code_all, 0)
        payload_default = json.loads(out_default)
        payload_all = json.loads(out_all)
        self.assertEqual(payload_default["mode"], payload_all["mode"])
        self.assertEqual(
            [r["docid"] for r in payload_default["results"]],
            [r["docid"] for r in payload_all["results"]],
        )

    def test_consolidated_text_mode_first_token_is_path_span(self) -> None:
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        first_line = out.splitlines()[0]
        first_token = first_line.split()[0]
        self.assertRegex(first_token, r"^[^:]+:\d+-\d+$")

    def test_consolidated_output_flag_writes_parseable_file(self) -> None:
        output_path = pathlib.Path(self._cache_tmp.name) / "handoff.json"
        code, _out = _run(
            [
                "query", "what carries data between networks",
                "--root", str(self.root), "--top-k", "1", "--output", str(output_path),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "consolidated")
        self.assertIn("results", payload)

    def test_consolidated_graceful_skip_exits_zero_when_extras_absent(self) -> None:
        # Even if every optional retriever is unavailable, the always-present
        # `lexical` strategy consolidating on its own must still exit 0.
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--top-k", "1", "--json"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("lexical", payload["retrievers"])

    def test_stats_reports_retriever_name(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        code, out = _run(["stats", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("retriever: lexical", out)

    def test_index_accepts_every_documented_retriever_choice(self) -> None:
        # Parser-level check only: unknown names must be rejected up front...
        with self.assertRaises(SystemExit):
            main(["index", "--root", str(self.root), "--retriever", "nope"])
        # ...while all documented choices parse (execution may still fail
        # later on missing optional extras, which is covered separately).
        for choice in (
            "lexical", "lexical+ctx", "turbovec", "pi-serini", "hybrid", "treesitter",
        ):
            with self.subTest(choice=choice):
                code, _out, _err = _run_with_stderr(
                    ["index", "--root", str(self.root), "--retriever", choice, "--force"]
                )
                self.assertIn(code, (0, 1))

    @unittest.skipIf(_TURBOVEC_INSTALLED, "turbovec installed; degradation not exercised")
    def test_index_hybrid_without_extras_fails_with_guidance(self) -> None:
        code, _out, err = _run_with_stderr(
            ["index", "--root", str(self.root), "--retriever", "hybrid"]
        )
        self.assertEqual(code, 1)
        self.assertIn("turbovec", err)

    @unittest.skipIf(_TURBOVEC_INSTALLED, "turbovec installed; degradation not exercised")
    def test_lexical_cache_does_not_satisfy_hybrid_index(self) -> None:
        code, _out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)
        # A fresh lexical cache must not short-circuit a hybrid index run:
        # hybrid has its own cache slot, so this attempts a real build and
        # fails on the missing extras instead of printing "up to date".
        code, out, err = _run_with_stderr(
            ["index", "--root", str(self.root), "--retriever", "hybrid"]
        )
        self.assertEqual(code, 1)
        self.assertNotIn("up to date", out)
        self.assertIn("turbovec", err)

    def test_index_default_builds_all_and_core_always_succeeds(self) -> None:
        code, out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("lexical: indexed 2 chunks", out)

        directory = index_dir(self.root)
        self.assertTrue((directory / "lexical.json").exists())
        self.assertTrue((directory / "meta.json").exists())

        for name in ("turbovec", "pi-serini", "hybrid"):
            with self.subTest(name=name):
                self.assertIn(name, out)

    @unittest.skipIf(_TURBOVEC_INSTALLED, "turbovec installed; degradation not exercised")
    def test_index_all_skips_missing_backends_without_hard_fail(self) -> None:
        code, out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("skipped", out)
        self.assertIn("turbovec", out)
        self.assertIn("hybrid: skipped", out)
        self.assertTrue((index_dir(self.root) / "lexical.json").exists())

    def test_index_accepts_all_choice(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "all", "--force"])
        self.assertEqual(code, 0)
        self.assertTrue((index_dir(self.root) / "lexical.json").exists())

    @unittest.skipUnless(_TURBOVEC_INSTALLED, "turbovec absent; all-mode can't fill every slot")
    def test_index_all_populates_every_slot(self) -> None:
        code, _out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)
        directory = index_dir(self.root)
        self.assertTrue((directory / "lexical.json").exists())
        self.assertTrue((directory / "turbovec.json").exists())
        self.assertTrue((directory / "hybrid.json").exists())
        # pi-serini additionally needs Java 21 + the pyserini extra; only
        # assert its cache file when it actually built successfully.
        pi_serini_cache = directory / "pi-serini.json"
        if pi_serini_cache.exists():
            self.assertTrue(pi_serini_cache.exists())

    @unittest.skipIf(_TURBOVEC_INSTALLED, "turbovec installed; degradation not exercised")
    def test_query_missing_backend_hard_fails_with_guidance(self) -> None:
        code, _out, err = _run_with_stderr(
            ["query", "x", "--root", str(self.root), "--retriever", "turbovec"]
        )
        self.assertEqual(code, 1)
        self.assertIn("turbovec", err)

    def test_treesitter_is_a_valid_retriever_choice(self) -> None:
        code, _out, err = _run_with_stderr(
            ["index", "--root", str(self.root), "--retriever", "treesitter", "--force"]
        )
        self.assertIn(code, (0, 1))
        if code == 1:
            self.assertIn("treesitter", err)

    @unittest.skipIf(
        _TREESITTER_INSTALLED, "tree-sitter-language-pack installed; skip path not exercised"
    )
    def test_index_all_skips_treesitter_when_extras_missing(self) -> None:
        # A real code file forces language_for_path -> chunk_code, whose guidance
        # RuntimeError is the skip signal when the treesitter extra is absent.
        _write(self.root / "sample.py", "def hello():\n    return 42\n")

        code, out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("treesitter: skipped", out)
        self.assertTrue((index_dir(self.root) / "lexical.json").exists())

    @unittest.skipUnless(_TREESITTER_INSTALLED, "tree-sitter-language-pack not installed")
    def test_index_all_builds_treesitter_when_installed(self) -> None:
        code, out = _run(["index", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("treesitter: indexed", out)
        self.assertTrue((index_dir(self.root) / "treesitter.json").exists())

    @unittest.skipUnless(_TREESITTER_INSTALLED, "tree-sitter-language-pack not installed")
    def test_query_treesitter_json_includes_context(self) -> None:
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "treesitter", "--force"]
        )
        self.assertEqual(code, 0)
        code, out = _run(
            [
                "query", "what carries data between networks",
                "--root", str(self.root), "--retriever", "treesitter",
                "--top-k", "1", "--json",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("context", payload["results"][0])

    def test_bad_args_exit_nonzero(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["not-a-real-command"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_required_query_arg_exits_nonzero(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["query"])
        self.assertNotEqual(ctx.exception.code, 0)

    # -- tune / --auto / hyperparameter flags ----------------------------------

    def test_tune_json_emits_signals_and_params(self) -> None:
        code, out = _run(["tune", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("signals", payload)
        self.assertIn("params", payload)
        self.assertIn("n_files", payload["signals"])
        self.assertIn("bm25_k1", payload["params"])
        self.assertIn("code_chars", payload["params"])

    def test_tune_writes_nothing(self) -> None:
        code, _out = _run(["tune", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        self.assertFalse(index_dir(self.root).exists())

    def test_bm25_k1_flag_reaches_persisted_meta(self) -> None:
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.9"]
        )
        self.assertEqual(code, 0)
        meta_path = index_dir(self.root) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["hyperparams"]["bm25_k1"], 1.9)

    def test_bm25_k1_flag_changes_search_behavior(self) -> None:
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.9"]
        )
        self.assertEqual(code, 0)
        directory = index_dir(self.root)
        data = json.loads((directory / "lexical.json").read_text(encoding="utf-8"))
        self.assertEqual(data["bm25"]["k1"], 1.9)

    def test_stale_rebuild_via_query_preserves_persisted_hyperparams(self) -> None:
        # Index with explicit, non-default hyperparameters.
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical",
             "--bm25-k1", "1.9", "--tokenizer", "code"]
        )
        self.assertEqual(code, 0)
        meta_path = index_dir(self.root) / "meta.json"
        first_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(first_meta["hyperparams"]["bm25_k1"], 1.9)
        self.assertEqual(first_meta["hyperparams"]["tokenizer"], "code")

        # Dirty the corpus fingerprint (new file), then query with NO flags —
        # this must trigger a query-side rebuild (not a fresh `index` run).
        new_file = self.root / "c.txt"
        _write(new_file, "asteroid belt lies between mars and jupiter")
        _bump_mtime(new_file)

        code, out = _run(
            ["query", "asteroid belt mars jupiter",
             "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "c.txt:1-1")

        # The rebuilt cache must still carry the original, explicitly-set
        # hyperparameters — not silently reverted to static defaults
        # (k1=1.5, tokenizer=plain) just because the query itself passed no
        # hyperparameter flags.
        rebuilt_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt_meta["hyperparams"]["bm25_k1"], 1.9)
        self.assertEqual(rebuilt_meta["hyperparams"]["tokenizer"], "code")
        rebuilt_data = json.loads(
            (index_dir(self.root) / "lexical.json").read_text(encoding="utf-8")
        )
        self.assertEqual(rebuilt_data["bm25"]["k1"], 1.9)
        self.assertEqual(rebuilt_data["bm25"]["tokenizer"], "code")

    def test_stale_rebuild_via_bare_index_preserves_persisted_hyperparams(self) -> None:
        # Index with explicit, non-default hyperparameters.
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical",
             "--bm25-k1", "1.9", "--tokenizer", "code"]
        )
        self.assertEqual(code, 0)
        meta_path = index_dir(self.root) / "meta.json"
        first_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(first_meta["hyperparams"]["bm25_k1"], 1.9)
        self.assertEqual(first_meta["hyperparams"]["tokenizer"], "code")

        # Dirty the corpus fingerprint, then a bare `index` (no flags at
        # all) — this is the routine "refresh my index" workflow.
        new_file = self.root / "c.txt"
        _write(new_file, "asteroid belt lies between mars and jupiter")
        _bump_mtime(new_file)

        code, out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        self.assertIn("indexed 3 chunks ->", out)

        rebuilt_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt_meta["hyperparams"]["bm25_k1"], 1.9)
        self.assertEqual(rebuilt_meta["hyperparams"]["tokenizer"], "code")
        rebuilt_data = json.loads(
            (index_dir(self.root) / "lexical.json").read_text(encoding="utf-8")
        )
        self.assertEqual(rebuilt_data["bm25"]["k1"], 1.9)
        self.assertEqual(rebuilt_data["bm25"]["tokenizer"], "code")

    def test_stale_rebuild_via_index_force_preserves_persisted_hyperparams(self) -> None:
        # --force means "rebuild", not "reset my settings": a bare --force
        # (no hyperparameter flags) must also recover recorded values.
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical",
             "--bm25-k1", "1.9", "--tokenizer", "code"]
        )
        self.assertEqual(code, 0)

        code, out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--force"]
        )
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 chunks ->", out)

        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["hyperparams"]["bm25_k1"], 1.9)
        self.assertEqual(meta["hyperparams"]["tokenizer"], "code")
        data = json.loads((index_dir(self.root) / "lexical.json").read_text(encoding="utf-8"))
        self.assertEqual(data["bm25"]["k1"], 1.9)
        self.assertEqual(data["bm25"]["tokenizer"], "code")

    def test_explicit_flag_still_overwrites_recorded_hyperparams_on_reindex(self) -> None:
        # Explicit flags/--auto must always win over recovery, even on a
        # from-existing-meta rebuild.
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical",
             "--bm25-k1", "1.9", "--tokenizer", "code"]
        )
        self.assertEqual(code, 0)

        code, out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical",
             "--force", "--bm25-k1", "1.1"]
        )
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 chunks ->", out)

        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["hyperparams"]["bm25_k1"], 1.1)
        # tokenizer was not re-specified this time, but since this run
        # passed *some* explicit flag, resolve_params was invoked fresh
        # from static defaults + that override — tokenizer reverts to
        # "plain" here, which is the documented "explicit run" semantics
        # (an explicit hyperparameter run always starts from static
        # defaults + overrides, never merges with prior recorded values).
        self.assertEqual(meta["hyperparams"]["tokenizer"], "plain")

    def test_from_scratch_index_with_no_flags_keeps_static_defaults(self) -> None:
        # No prior meta to recover from: must behave exactly as before this
        # fix (no hyperparams/corpus_stats block at all).
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertNotIn("hyperparams", meta)
        self.assertNotIn("corpus_stats", meta)

    def test_auto_index_then_plain_query_reuses_persisted_tokenizer(self) -> None:
        code, out = _run(["index", "--root", str(self.root), "--retriever", "lexical", "--auto"])
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 chunks ->", out)

        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertIn("hyperparams", meta)
        self.assertIn("corpus_stats", meta)
        self.assertTrue(meta["corpus_stats"]["auto"])
        persisted_tokenizer = meta["hyperparams"]["tokenizer"]

        data = json.loads((index_dir(self.root) / "lexical.json").read_text(encoding="utf-8"))
        self.assertEqual(data["bm25"]["tokenizer"], persisted_tokenizer)

        # A plain query afterward must reuse the persisted tokenizer, not
        # re-derive it — same-cache round trip, no --auto needed to query.
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt:1-1")

    def test_no_hyperparameter_flags_omits_meta_blocks(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertNotIn("hyperparams", meta)
        self.assertNotIn("corpus_stats", meta)

    def test_changed_bm25_k1_triggers_rebuild_same_value_reports_up_to_date(self) -> None:
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.2"]
        )
        self.assertEqual(code, 0)
        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns

        # Same value again: fast path, no rewrite.
        code, out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.2"]
        )
        self.assertEqual(code, 0)
        self.assertIn("up to date", out)
        self.assertEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

        # Changed value: triggers a rebuild despite an unchanged corpus.
        code, out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.7"]
        )
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 chunks ->", out)
        self.assertNotEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

    def test_query_with_persisted_nondefault_extensions_performs_no_rebuild(self) -> None:
        # A cache built (e.g. by a prior CLI version, or directly via
        # persistence.save_index) with a non-default `extensions` loader_kw
        # must be recognized as up to date by a plain `query` — it should
        # rehydrate the loader kwargs from the cache's own meta rather than
        # falling back to discover_files' defaults (which would omit the
        # .dat file and spuriously look stale).
        _write(self.root / "note.dat", "asteroid belt lies between mars and jupiter")
        loader_kw = {"extensions": frozenset({".txt", ".dat"})}
        fingerprint = compute_fingerprint(self.root, **loader_kw)
        retriever = LexicalRetriever()
        retriever.index(load_chunk_documents(self.root, **loader_kw))
        save_index(
            retriever, self.root, fingerprint, "lexical", "0.7.0", loader_kw=loader_kw
        )
        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns

        code, out = _run(
            ["query", "asteroid belt mars jupiter",
             "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "note.dat:1-1")
        self.assertEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

    def test_stats_echoes_hyperparams_and_corpus_stats(self) -> None:
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.3"]
        )
        self.assertEqual(code, 0)
        code, out = _run(["stats", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("hyperparams:", out)

    # -- lexical+ctx large-corpus guard (T-C6) ---------------------------------
    #
    # ``lexical+ctx`` issues one LLM call per chunk-Document, so exercising
    # the "passes with --allow-large-context" / "small corpus unaffected"
    # paths through a real network call isn't safe in a test environment
    # without an ANTHROPIC_API_KEY; ``_check_lexical_ctx_guard`` (the pure
    # decision function ``_cmd_index`` calls before ``retriever.index()``)
    # is exercised directly instead, plus one full-CLI test proving the
    # guard actually blocks ``index --retriever lexical+ctx`` end-to-end
    # (via monkeypatched thresholds so it never has to synthesize 2000+
    # real chunks).

    def test_lexical_ctx_guard_raises_above_hard_limit_without_flag(self) -> None:
        documents = [Document(f"d{i}", "x") for i in range(3)]
        with mock.patch.object(cli, "_LEXICAL_CTX_HARD_LIMIT_CHUNKS", 2):
            with self.assertRaises(RuntimeError) as ctx:
                cli._check_lexical_ctx_guard("lexical+ctx", documents, False)
        self.assertIn("--allow-large-context", str(ctx.exception))

    def test_lexical_ctx_guard_allows_above_hard_limit_with_flag(self) -> None:
        documents = [Document(f"d{i}", "x") for i in range(3)]
        with mock.patch.object(cli, "_LEXICAL_CTX_HARD_LIMIT_CHUNKS", 2):
            cli._check_lexical_ctx_guard("lexical+ctx", documents, True)  # must not raise

    def test_lexical_ctx_guard_warns_above_warn_threshold_below_hard_limit(self) -> None:
        documents = [Document(f"d{i}", "x") for i in range(3)]
        with mock.patch.object(cli, "_LEXICAL_CTX_WARN_CHUNKS", 2), mock.patch.object(
            cli, "_LEXICAL_CTX_HARD_LIMIT_CHUNKS", 10
        ):
            err = io.StringIO()
            with redirect_stderr(err):
                cli._check_lexical_ctx_guard("lexical+ctx", documents, False)
        self.assertIn("warning", err.getvalue())

    def test_lexical_ctx_guard_small_corpus_unaffected(self) -> None:
        documents = [Document("d0", "x"), Document("d1", "y")]
        err = io.StringIO()
        with redirect_stderr(err):
            cli._check_lexical_ctx_guard("lexical+ctx", documents, False)  # must not raise
        self.assertEqual(err.getvalue(), "")

    def test_lexical_ctx_guard_is_a_noop_for_other_retrievers(self) -> None:
        documents = [Document(f"d{i}", "x") for i in range(3)]
        with mock.patch.object(cli, "_LEXICAL_CTX_HARD_LIMIT_CHUNKS", 2):
            cli._check_lexical_ctx_guard("lexical", documents, False)  # must not raise

    def test_index_lexical_ctx_blocked_end_to_end_without_flag(self) -> None:
        # Full _cmd_index plumbing: the guard fires before retriever.index()
        # is ever reached, so this needs no network access.
        with mock.patch.object(cli, "_LEXICAL_CTX_HARD_LIMIT_CHUNKS", 1):
            code, _out, err = _run_with_stderr(
                ["index", "--root", str(self.root), "--retriever", "lexical+ctx"]
            )
        self.assertEqual(code, 1)
        self.assertIn("--allow-large-context", err)
        # Confirm no cache was ever written (the guard raised before index()).
        self.assertFalse((index_dir(self.root) / "lexical.json").exists())

    def test_index_lexical_ctx_all_mode_never_triggers_guard(self) -> None:
        # lexical+ctx is deliberately excluded from _DEFAULT_INDEX_SET, so
        # the 'all' path must never even consult the guard.
        with mock.patch.object(cli, "_LEXICAL_CTX_HARD_LIMIT_CHUNKS", 0):
            code, _out = _run(["index", "--root", str(self.root), "--force"])
        self.assertEqual(code, 0)


class TestCliPdfAutoActivation(unittest.TestCase):
    """CLI-surface coverage for auto-activated PDF indexing (WP-B): the
    ``--no-pdf`` escape hatch, the ``extract`` subcommand, and the
    end-to-end index/query path over a PDF-containing tree."""

    def setUp(self) -> None:
        self._project_tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._project_tmp.name)
        _write(self.root / "a.txt", "routers forward packets between networks and carry data")

        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        os.environ["RETRIEVAL_INDEX_DIR"] = self._cache_tmp.name
        extractors.clear_process_cache()
        extractors._WARNED = False

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        else:
            os.environ["RETRIEVAL_INDEX_DIR"] = self._old_env
        self._cache_tmp.cleanup()
        self._project_tmp.cleanup()
        extractors.clear_process_cache()
        extractors._WARNED = False

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_index_then_query_on_pdf_tree_is_stable_and_cites_sidecar(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        _write_pdf(pdf_path, pages=2)

        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)
        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns
        first_created_at = json.loads(meta_path.read_text())["created_at"]

        code, out = _run(
            [
                "query", "wonderful indeed spanning multiple lines reflow logic",
                "--root", str(self.root), "--retriever", "lexical", "--top-k", "1",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(".agentic-retrieval/extracted/docs/paper.pdf.md", out)
        # No rebuild: meta is untouched by the flag-less query.
        self.assertEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)
        self.assertEqual(json.loads(meta_path.read_text())["created_at"], first_created_at)

    @unittest.skipIf(_PYPDF_INSTALLED, "pypdf installed; degradation path not exercised")
    def test_index_succeeds_without_pypdf_with_one_warning_and_stub(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        code, out, err = _run_with_stderr(
            ["index", "--root", str(self.root), "--retriever", "lexical"]
        )
        self.assertEqual(code, 0)
        self.assertIn("indexed", out)
        self.assertEqual(err.count("pypdf is not installed"), 1)

        manifest = extractors.load_manifest(self.root)
        entry = manifest["entries"]["docs/paper.pdf"]
        self.assertEqual(entry["status"], "stub")
        self.assertEqual(entry["reason"], "backend-missing")

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_extraction_happens_once_per_pdf_under_index_auto(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        _write_pdf(pdf_path, pages=2)

        counter = {"n": 0}
        original = extractors._EXTRACTORS[".pdf"]

        def counting_extractor(data: bytes):
            counter["n"] += 1
            return original(data)

        extractors._EXTRACTORS[".pdf"] = counting_extractor
        try:
            code, _out = _run(["index", "--root", str(self.root), "--auto"])
        finally:
            extractors._EXTRACTORS[".pdf"] = original
        self.assertEqual(code, 0)
        self.assertEqual(counter["n"], 1)

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_query_json_on_pdf_corpus_includes_page_breadcrumb(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        _write_pdf(pdf_path, pages=2)

        code, out = _run(
            [
                "query", "wonderful indeed spanning multiple lines reflow logic",
                "--root", str(self.root), "--retriever", "lexical",
                "--top-k", "1", "--json",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("Page", payload["results"][0]["context"])

    def test_no_pdf_flag_excludes_pdfs_and_is_sticky(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--no-pdf"]
        )
        self.assertEqual(code, 0)
        sidecar_dir = self.root / ".agentic-retrieval" / "extracted"
        self.assertFalse(sidecar_dir.exists())

        meta_path = index_dir(self.root) / "meta.json"
        first_mtime_ns = meta_path.stat().st_mtime_ns

        # Flag-less query afterward: --no-pdf is sticky via the persisted
        # meta, so this must not rebuild (and must not touch the PDF).
        code, _out = _run(
            [
                "query", "routers forward packets", "--root", str(self.root),
                "--retriever", "lexical", "--top-k", "1",
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)
        self.assertFalse(sidecar_dir.exists())

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_extract_subcommand_creates_sidecars_force_and_prune(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        _write_pdf(pdf_path, pages=2)

        code, out = _run(["extract", "--root", str(self.root)])
        self.assertEqual(code, 0)
        sidecar_path = (
            self.root / ".agentic-retrieval" / "extracted" / "docs" / "paper.pdf.md"
        )
        self.assertTrue(sidecar_path.exists())
        self.assertIn(
            "docs/paper.pdf -> .agentic-retrieval/extracted/docs/paper.pdf.md", out
        )
        self.assertIn("pages", out)
        first_mtime_ns = sidecar_path.stat().st_mtime_ns

        time.sleep(0.01)
        code, out = _run(["extract", "--root", str(self.root), "--force"])
        self.assertEqual(code, 0)
        self.assertNotEqual(sidecar_path.stat().st_mtime_ns, first_mtime_ns)

        pdf_path.unlink()
        code, out = _run(["extract", "--root", str(self.root), "--prune"])
        self.assertEqual(code, 0)
        self.assertIn("pruned 1", out)
        self.assertFalse(sidecar_path.exists())

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_extract_json_summary(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        _write_pdf(pdf_path, pages=1)

        code, out = _run(["extract", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload["files"]), 1)
        self.assertEqual(payload["files"][0]["source"], "docs/paper.pdf")
        self.assertEqual(payload["files"][0]["status"], "ok")
        self.assertEqual(payload["pruned"], 0)

    @unittest.skipIf(_PYPDF_INSTALLED, "pypdf installed; guidance path not exercised")
    def test_extract_without_pypdf_exits_nonzero_with_guidance(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        code, _out, err = _run_with_stderr(["extract", "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertEqual(err.count("error:"), 1)
        self.assertIn(".[pdf]", err)

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_extract_prune_does_not_delete_registered_non_pdf_media_sidecar(self) -> None:
        docx_path = self.root / "report.docx"
        docx_path.write_bytes(b"not a real docx payload")
        extractors.register_sidecar(self.root, docx_path, "Hand-authored docx transcript.")
        sidecar_path = self.root / ".agentic-retrieval" / "extracted" / "report.docx.md"
        self.assertTrue(sidecar_path.exists())

        code, out = _run(["extract", "--root", str(self.root), "--prune"])
        self.assertEqual(code, 0)
        self.assertNotIn("pruned 1", out)
        self.assertTrue(sidecar_path.exists())

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_extract_does_not_attempt_non_pdf_media(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        _write_pdf(pdf_path, pages=1)
        docx_path = self.root / "report.docx"
        docx_path.write_bytes(b"not a real docx payload")

        code, out = _run(["extract", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload["files"]), 1)
        self.assertEqual(payload["files"][0]["source"], "docs/paper.pdf")
        # extract never touched the .docx: no sidecar/manifest entry exists.
        docx_sidecar = self.root / ".agentic-retrieval" / "extracted" / "report.docx.md"
        self.assertFalse(docx_sidecar.exists())

    def test_extract_works_on_captions_only_tree_without_pypdf(self) -> None:
        # require_extractors is now called with the *discovered* suffixes,
        # not the blanket EXTRACTABLE_EXTENSIONS, so a caption-only tree
        # must succeed even when pypdf isn't installed.
        srt_path = self.root / "talk.srt"
        srt_path.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nHello from a caption file.\n",
            encoding="utf-8",
        )
        code, out = _run(["extract", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload["files"]), 1)
        self.assertEqual(payload["files"][0]["source"], "talk.srt")
        self.assertEqual(payload["files"][0]["status"], "ok")


class TestSidecarCommand(unittest.TestCase):
    """CLI-surface coverage for the no-pypdf 'sidecar' subcommand: --list
    and --register never import/require pypdf (contrast with 'extract',
    which hard-fails without it)."""

    def setUp(self) -> None:
        self._project_tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._project_tmp.name)
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        os.environ["RETRIEVAL_INDEX_DIR"] = self._cache_tmp.name
        extractors.clear_process_cache()
        extractors._WARNED = False

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        else:
            os.environ["RETRIEVAL_INDEX_DIR"] = self._old_env
        self._cache_tmp.cleanup()
        self._project_tmp.cleanup()
        extractors.clear_process_cache()
        extractors._WARNED = False

    def test_sidecar_list_reports_stub_state_without_pypdf(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        code, out = _run(["sidecar", "--list", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("docs/paper.pdf: missing", out)

        original = extractors.backend_available
        extractors.backend_available = lambda: False
        try:
            extractors.ensure_sidecar(self.root, pdf_path)
            code, out = _run(["sidecar", "--list", "--root", str(self.root)])
        finally:
            extractors.backend_available = original
        self.assertEqual(code, 0)
        self.assertIn("docs/paper.pdf: stub (backend-missing)", out)

    def test_sidecar_list_json_shape(self) -> None:
        pdf_path = self.root / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        code, out = _run(["sidecar", "--list", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["root"], str(self.root.resolve()))
        self.assertIn("backend_available", payload)
        self.assertEqual(len(payload["files"]), 1)
        self.assertEqual(payload["files"][0]["rel"], "paper.pdf")
        self.assertEqual(payload["files"][0]["state"], "missing")

    def test_sidecar_list_reports_agent_authored_after_register(self) -> None:
        pdf_path = self.root / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
        extractors.register_sidecar(self.root, pdf_path, "hand-authored transcript")

        code, out = _run(["sidecar", "--list", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["files"][0]["state"], "agent-authored")

    def test_sidecar_register_from_file(self) -> None:
        pdf_path = self.root / "docs" / "paper.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
        transcript_path = self.root / "transcript.md"
        transcript_path.write_text("## Page 1\n\nHand-authored content.", encoding="utf-8")

        code, out = _run(
            [
                "sidecar", "--register", str(pdf_path),
                "--transcript", str(transcript_path), "--root", str(self.root),
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("docs/paper.pdf ->", out)
        sidecar_path = self.root / ".agentic-retrieval" / "extracted" / "docs" / "paper.pdf.md"
        self.assertTrue(sidecar_path.exists())
        self.assertIn("Hand-authored content.", sidecar_path.read_text(encoding="utf-8"))

    def test_sidecar_register_from_stdin(self) -> None:
        pdf_path = self.root / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch(
            "sys.stdin", io.StringIO("## Page 1\n\nFrom stdin.")
        ):
            code = main(
                [
                    "sidecar", "--register", str(pdf_path),
                    "--transcript", "-", "--root", str(self.root),
                ]
            )
        self.assertEqual(code, 0)
        sidecar_path = self.root / ".agentic-retrieval" / "extracted" / "paper.pdf.md"
        self.assertIn("From stdin.", sidecar_path.read_text(encoding="utf-8"))

    def test_sidecar_register_missing_source_exits_nonzero_with_guidance(self) -> None:
        # Transcript comes from a real file (not stdin) here: the point
        # under test is the missing-source error path, and reading actual
        # process stdin in-process would block the test run indefinitely.
        transcript_path = self.root / "transcript.md"
        transcript_path.write_text("## Page 1\n\ncontent", encoding="utf-8")
        code, _out, err = _run_with_stderr(
            [
                "sidecar", "--register", str(self.root / "nope.pdf"),
                "--transcript", str(transcript_path), "--root", str(self.root),
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_sidecar_requires_list_or_register(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                main(["sidecar", "--root", str(self.root)])

    def test_query_auto_reindexes_after_sidecar_registration(self) -> None:
        _write(self.root / "a.txt", "routers forward packets between networks")
        pdf_path = self.root / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")

        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        extractors.register_sidecar(
            self.root, pdf_path, "## Page 1\n\nA distinctive elephant migration pattern."
        )

        # Bare query, no --force: registration changed the fingerprint (via
        # agent_sidecar_revision), so this must auto-reindex and surface a
        # hit from the newly registered transcript.
        code, out = _run(
            [
                "query", "distinctive elephant migration pattern",
                "--root", str(self.root), "--retriever", "lexical", "--top-k", "1",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(".agentic-retrieval/extracted/paper.pdf.md", out)

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_extract_force_warns_before_overwriting_agent_sidecar(self) -> None:
        pdf_path = self.root / "paper.pdf"
        _write_pdf(pdf_path, pages=1)
        extractors.register_sidecar(self.root, pdf_path, "hand-authored transcript")

        code, _out, err = _run_with_stderr(["extract", "--root", str(self.root), "--force"])
        self.assertEqual(code, 0)
        self.assertIn("warning: overwriting 1 agent-authored sidecar(s)", err)

    def test_sidecar_list_reports_docx_as_agent_only_stub(self) -> None:
        docx_path = self.root / "report.docx"
        docx_path.write_bytes(b"not a real docx payload")

        code, out = _run(["sidecar", "--list", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("report.docx: missing", out)

        extractors.ensure_sidecar(self.root, docx_path)
        code, out = _run(["sidecar", "--list", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("report.docx: stub (agent-only)", out)

    def test_sidecar_register_and_query_end_to_end_on_docx(self) -> None:
        _write(self.root / "a.txt", "routers forward packets between networks")
        docx_path = self.root / "report.docx"
        docx_path.write_bytes(b"not a real docx payload")

        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        transcript_path = self.root / "transcript.md"
        transcript_path.write_text(
            "## Page 1\n\nA distinctive elephant migration pattern.", encoding="utf-8"
        )
        code, out = _run(
            [
                "sidecar", "--register", str(docx_path),
                "--transcript", str(transcript_path), "--root", str(self.root),
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("report.docx ->", out)

        # Bare query, no --force: registration changed the fingerprint, so
        # this must auto-reindex and surface a hit from the transcript.
        code, out = _run(
            [
                "query", "distinctive elephant migration pattern",
                "--root", str(self.root), "--retriever", "lexical", "--top-k", "1",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(".agentic-retrieval/extracted/report.docx.md", out)

    def test_sidecar_list_shows_mp4_as_agent_orchestrated_stub(self) -> None:
        mp4_path = self.root / "recording.mp4"
        mp4_path.write_bytes(b"fake video payload")

        code, out = _run(["sidecar", "--list", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("recording.mp4: missing", out)

        extractors.ensure_sidecar(self.root, mp4_path)
        code, out = _run(["sidecar", "--list", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("recording.mp4: stub (agent-orchestrated)", out)

    def test_sidecar_register_on_mp4_reports_section_count(self) -> None:
        mp4_path = self.root / "recording.mp4"
        mp4_path.write_bytes(b"fake video payload")
        transcript_path = self.root / "transcript.md"
        transcript_path.write_text(
            "## [00:00:00] Intro\n\n[00:00:00] Hello from an ASR transcript.\n\n"
            "## [00:03:15] Middle\n\n[00:03:15] A later section of the talk.",
            encoding="utf-8",
        )
        code, out = _run(
            [
                "sidecar", "--register", str(mp4_path),
                "--transcript", str(transcript_path), "--root", str(self.root),
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("recording.mp4 ->", out)
        self.assertIn("2 pages", out)

    def test_sidecar_register_zero_pages_note_is_truthful_not_breadcrumb_absent(
        self,
    ) -> None:
        # A non-paged format (here a fake .pptx) has no '## Page N'
        # headings, so the manifest's `pages` counter is 0 - but section
        # breadcrumbs still come from the '## Slide N' heading, so the
        # stderr note must not claim breadcrumbs are absent.
        pptx_path = self.root / "deck.pptx"
        pptx_path.write_bytes(b"not a real pptx payload")
        transcript_path = self.root / "transcript.md"
        transcript_path.write_text(
            "## Slide 1 — Intro\n\nWelcome slide content.", encoding="utf-8"
        )

        code, _out, err = _run_with_stderr(
            [
                "sidecar", "--register", str(pptx_path),
                "--transcript", str(transcript_path), "--root", str(self.root),
            ]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("breadcrumbs will be absent", err)
        self.assertIn("0 pages", err)

    def test_sidecar_register_zero_pages_sidecar_context_is_slide_heading(self) -> None:
        pptx_path = self.root / "deck.pptx"
        pptx_path.write_bytes(b"not a real pptx payload")
        transcript_path = self.root / "transcript.md"
        transcript_path.write_text(
            "## Slide 1 — Intro\n\nWelcome slide content.", encoding="utf-8"
        )

        code, _out = _run(
            [
                "sidecar", "--register", str(pptx_path),
                "--transcript", str(transcript_path), "--root", str(self.root),
            ]
        )
        self.assertEqual(code, 0)

        documents = load_chunk_documents(self.root)
        sidecar_docid = ".agentic-retrieval/extracted/deck.pptx.md"
        doc = next(
            d for d in documents
            if d.source_path == sidecar_docid and "Welcome slide content" in d.text
        )
        self.assertEqual(doc.context, "Slide 1 — Intro")


class TestCliPoTranslationMemory(unittest.TestCase):
    def setUp(self) -> None:
        self._project_tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._project_tmp.name)
        shutil.copy(_ROOT_DIR / "tests" / "fixtures" / "po" / "sample.po", self.root / "sample.po")
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        os.environ["RETRIEVAL_INDEX_DIR"] = self._cache_tmp.name

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        else:
            os.environ["RETRIEVAL_INDEX_DIR"] = self._old_env
        self._cache_tmp.cleanup()
        self._project_tmp.cleanup()

    def _index(self) -> str:
        code, out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--tokenizer", "cjk"]
        )
        self.assertEqual(code, 0)
        return out

    def _query_json(self, query: str, *extra: str):
        code, out = _run(
            ["query", query, "--root", str(self.root), "--retriever", "lexical",
             "--top-k", "1", "--json", *extra]
        )
        self.assertEqual(code, 0)
        return json.loads(out)["results"][0]

    def test_index_reports_entries_and_persists_tokenizer(self) -> None:
        self.assertIn("indexed 6 chunks ->", self._index())
        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["hyperparams"]["tokenizer"], "cjk")

    def test_chinese_query_surfaces_msgid_msgstr(self) -> None:
        self._index()
        hit = self._query_json("疊代器")
        self.assertEqual(hit["path"], "sample.po")
        self.assertEqual(hit["start_line"], 8)
        self.assertEqual(hit["metadata"]["msgid"], "Iterator Types")
        self.assertIn("疊代器", hit["metadata"]["msgstr"])

    def test_english_query_hits_untranslated_entry(self) -> None:
        self._index()
        hit = self._query_json("logging configuration root logger")
        self.assertEqual(hit["start_line"], 25)
        self.assertIs(hit["metadata"]["translated"], False)

    def test_mixed_query_hits_fuzzy_entry(self) -> None:
        self._index()
        hit = self._query_json("frozenset 物件")
        self.assertEqual(hit["start_line"], 20)
        self.assertIs(hit["metadata"]["fuzzy"], True)

    def test_text_mode_prints_msgid_and_msgstr(self) -> None:
        self._index()
        code, out = _run(
            ["query", "疊代器", "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "sample.po:8-10")
        self.assertEqual(lines[1], "    msgid: Iterator Types")
        self.assertEqual(lines[2], "    msgstr: 疊代器型別")

    def test_sticky_tokenizer_on_stale_rebuild(self) -> None:
        self._index()
        _bump_mtime(self.root / "sample.po")
        code, _out = _run(
            ["query", "疊代器", "--root", str(self.root), "--retriever", "lexical", "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        meta = json.loads((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["hyperparams"]["tokenizer"], "cjk")
        data = json.loads((index_dir(self.root) / "lexical.json").read_text(encoding="utf-8"))
        self.assertEqual(data["bm25"]["tokenizer"], "cjk")

    def test_consolidated_query_includes_metadata(self) -> None:
        self._index()
        code, out = _run(
            ["query", "疊代器", "--root", str(self.root), "--json", "--top-k", "3"]
        )
        self.assertEqual(code, 0)
        results = json.loads(out)["results"]
        match = [r for r in results if r["path"] == "sample.po" and r["start_line"] == 8]
        self.assertTrue(match)
        self.assertEqual(match[0]["metadata"]["msgid"], "Iterator Types")

    def test_tune_json_reports_cjk_fraction(self) -> None:
        code, out = _run(["tune", "--root", str(self.root), "--json"])
        self.assertEqual(code, 0)
        self.assertGreaterEqual(json.loads(out)["signals"]["cjk_fraction"], 0.1)


class TestCliRecordDetails(unittest.TestCase):
    """Text-mode detail lines for a non-PO record kind (registered fake loader)."""

    def setUp(self) -> None:
        register_fake_xyz(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def _query_text(self, metadata) -> list:
        hit = SearchHit(
            docid="data.xyz:1-1",
            source_path="data.xyz",
            start_line=1,
            end_line=1,
            rank=1,
            metadata=metadata,
        )
        stub = mock.Mock()
        stub.search_detailed.return_value = [hit]
        with mock.patch.object(cli, "_load_or_rebuild", return_value=stub):
            code, out = _run(
                ["query", "hello", "--root", self.root, "--retriever", "lexical", "--top-k", "1"]
            )
        self.assertEqual(code, 0)
        return out.splitlines()

    def test_consolidated_text_renders_generic_details(self) -> None:
        hit = ConsolidatedHit(
            source_path="data.xyz",
            start_line=1,
            end_line=1,
            docid="data.xyz:1-1",
            context="",
            score=0.5,
            rank=1,
            provenance=["lexical"],
            metadata={"kind": "xyz_record", "title": "Hello", "lang": "en", "extra": "x"},
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._print_consolidated_text(
                {"skipped": [], "retrievers": ["lexical"], "results": [hit]}
            )
        self.assertEqual(buf.getvalue().splitlines()[1:], ["    title: Hello", "    lang: en"])

    def test_single_retriever_text_renders_generic_details(self) -> None:
        lines = self._query_text({"kind": "xyz_record", "title": "Hello", "lang": "en"})
        self.assertEqual(lines, ["data.xyz:1-1", "    title: Hello", "    lang: en"])

    def test_hit_without_metadata_prints_no_details(self) -> None:
        self.assertEqual(self._query_text({}), ["data.xyz:1-1"])


class TestIndexPerf(unittest.TestCase):
    """Redundant-work removal in the index/query path."""

    def setUp(self) -> None:
        self._project_tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._project_tmp.name)
        _write(self.root / "a.txt", "routers forward packets between networks and carry data")
        _write(self.root / "b.txt", "photosynthesis converts sunlight into chemical energy")
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        os.environ["RETRIEVAL_INDEX_DIR"] = self._cache_tmp.name
        self.dense_arms: list = []
        real_make = cli._make_retriever

        def fake_make(root, name, params=None):
            retriever = real_make(root, name, params)
            if name == "turbovec":
                dense = _StubDense(retriever._model_name, retriever._bit_width)
                self.dense_arms.append(dense)
                return dense
            if name == "hybrid":
                dense = _StubDense(retriever._dense._model_name, retriever._dense._bit_width)
                self.dense_arms.append(dense)
                retriever._dense = dense
            return retriever

        patcher = mock.patch("retrieval.cli._make_retriever", side_effect=fake_make)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("RETRIEVAL_INDEX_DIR", None)
        else:
            os.environ["RETRIEVAL_INDEX_DIR"] = self._old_env
        self._cache_tmp.cleanup()
        self._project_tmp.cleanup()

    @property
    def index_calls(self) -> int:
        return sum(d.index_calls for d in self.dense_arms)

    def _read(self, name: str):
        return json.loads((index_dir(self.root) / name).read_text(encoding="utf-8"))

    def _index(self, *extra: str):
        return _run(["index", "--root", str(self.root), *extra])

    def test_up_to_date_check_never_loads_data_file(self) -> None:
        self._index("--retriever", "lexical", "--force")
        with mock.patch("retrieval.cli.load_index", side_effect=AssertionError("heavy load")):
            code, out = self._index("--retriever", "lexical")
        self.assertEqual(code, 0)
        self.assertIn("up to date", out)

    def test_up_to_date_all_never_loads_data_file(self) -> None:
        self._index("--force")
        # Stubbed retrievers don't need real optional backends installed.
        with mock.patch("retrieval.persistence._modules_available", return_value=True), \
                mock.patch("retrieval.cli.load_index", side_effect=AssertionError("heavy load")):
            code, out = self._index()
        self.assertEqual(code, 0)
        self.assertIn("lexical: up to date", out)
        self.assertIn("hybrid: up to date", out)

    def test_corrupt_lexical_data_is_not_up_to_date(self) -> None:
        self._index("--retriever", "lexical", "--force")
        path = index_dir(self.root) / "lexical.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema"] = 2
        path.write_text(json.dumps(data), encoding="utf-8")
        _code, out = self._index("--retriever", "lexical")
        self.assertIn("indexed", out)
        self.assertNotIn("up to date", out)

    def test_index_all_reuses_built_arms_for_hybrid(self) -> None:
        code, out = self._index("--force")
        self.assertEqual(code, 0)
        self.assertEqual(self.index_calls, 1)
        self.assertIn("hybrid: indexed 2 chunks", out)
        hybrid = self._read("hybrid.json")
        self.assertEqual(hybrid["lexical"], self._read("lexical.json"))
        self.assertEqual(hybrid["dense"], self._read("turbovec.json"))

    def test_hybrid_reuse_output_matches_full_build(self) -> None:
        self._index("--force")
        reused = self._read("hybrid.json")
        calls_after_all = self.index_calls
        with tempfile.TemporaryDirectory() as other:
            os.environ["RETRIEVAL_INDEX_DIR"] = other
            self.dense_arms.clear()
            self._index("--retriever", "hybrid", "--force")
            full = self._read("hybrid.json")
        self.assertEqual(self.index_calls, 1)
        self.assertEqual(calls_after_all, 1)
        self.assertEqual(reused, full)
        self.assertEqual(list(reused), list(full))

    def test_index_all_partial_reuse_when_lexical_up_to_date(self) -> None:
        self._index("--retriever", "lexical", "--force")
        self._index()
        self.assertEqual(self.index_calls, 1)
        self.assertEqual(self._read("hybrid.json")["lexical"], self._read("lexical.json"))

    def test_index_all_hybrid_falls_back_when_arm_params_differ(self) -> None:
        self._index("--force", "--bit-width", "2")
        # hybrid built with different dense params than a default-built
        # turbovec arm would reuse: seed turbovec at 4 bits, then rebuild all
        # with hybrid at 2 bits via a fresh run.
        self.dense_arms.clear()
        real = cli._make_retriever  # the patched factory
        calls = []

        def mismatched(root, name, params=None):
            retriever = real(root, name, params)
            calls.append(name)
            if name == "hybrid":
                retriever._dense._bit_width = 8  # arms no longer compatible
            return retriever

        with mock.patch("retrieval.cli._make_retriever", side_effect=mismatched):
            self._index("--force", "--bit-width", "2")
        self.assertEqual(self.index_calls, 2)

    def test_index_all_hybrid_skipped_when_turbovec_skipped(self) -> None:
        def boom(self, documents):
            raise RuntimeError("turbovec retriever needs extras")

        with mock.patch.object(_StubDense, "index", boom):
            code, out = self._index("--force")
        self.assertEqual(code, 0)
        self.assertIn("turbovec: skipped", out)
        self.assertIn("hybrid: skipped", out)

    def test_index_all_loads_line_documents_once(self) -> None:
        with mock.patch(
            "retrieval.cli.load_chunk_documents", wraps=load_chunk_documents
        ) as spy:
            self._index("--force")
        self.assertEqual(spy.call_count, 1)

    def test_index_all_computes_fingerprint_once(self) -> None:
        with mock.patch(
            "retrieval.cli.compute_fingerprint", wraps=compute_fingerprint
        ) as spy_cli, mock.patch(
            "retrieval.persistence.compute_fingerprint", wraps=compute_fingerprint
        ) as spy_persist:
            self._index("--force")
            self.assertEqual(spy_cli.call_count + spy_persist.call_count, 1)
            self._index()
            self.assertEqual(spy_cli.call_count + spy_persist.call_count, 2)

    def test_index_single_counts_without_second_to_dict(self) -> None:
        with mock.patch.object(
            LexicalRetriever, "to_dict", autospec=True, side_effect=LexicalRetriever.to_dict
        ) as spy:
            _code, out = self._index("--retriever", "lexical", "--force")
        self.assertEqual(spy.call_count, 1)
        self.assertIn("indexed 2 chunks ->", out)

    def test_query_loads_index_once(self) -> None:
        self._index("--retriever", "lexical", "--force")
        with mock.patch("retrieval.cli.load_index", wraps=cli.load_index) as spy:
            code, _out = _run(["query", "routers", "--root", str(self.root),
                               "--retriever", "lexical"])
        self.assertEqual(code, 0)
        self.assertEqual(spy.call_count, 1)

    def test_query_stale_cache_rebuilds_without_loading(self) -> None:
        self._index("--retriever", "lexical", "--force")
        old = self._read("meta.json")["fingerprint"]
        _bump_mtime(self.root / "a.txt")
        with mock.patch("retrieval.cli.load_index", wraps=cli.load_index) as spy:
            code, _out = _run(["query", "routers", "--root", str(self.root),
                               "--retriever", "lexical"])
        self.assertEqual(code, 0)
        self.assertEqual(spy.call_count, 0)
        self.assertNotEqual(self._read("meta.json")["fingerprint"], old)


if __name__ == "__main__":
    unittest.main()
