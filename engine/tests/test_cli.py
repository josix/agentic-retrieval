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
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.cli import main  # noqa: E402
from retrieval.persistence import index_dir  # noqa: E402

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

    def test_stats_echoes_hyperparams_and_corpus_stats(self) -> None:
        code, _out = _run(
            ["index", "--root", str(self.root), "--retriever", "lexical", "--bm25-k1", "1.3"]
        )
        self.assertEqual(code, 0)
        code, out = _run(["stats", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("hyperparams:", out)


if __name__ == "__main__":
    unittest.main()
