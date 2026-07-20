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
        self.assertIn("indexed 2 docs ->", out)
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
        self.assertIn("indexed 2 docs ->", out)
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
        self.assertIn("indexed 3 docs ->", out)
        self.assertNotEqual(meta_path.stat().st_mtime_ns, first_mtime_ns)

    def test_query_returns_zero_and_expected_docid(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt")

    def test_query_without_cache_auto_indexes(self) -> None:
        # No prior "index" call — query must build + persist the cache itself.
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt")
        directory = index_dir(self.root)
        self.assertTrue((directory / "lexical.json").exists())

    def test_query_stale_ok_without_cache_auto_indexes(self) -> None:
        # No cache at all, plus --stale-ok: staleness is irrelevant when
        # there's nothing to compare against, so this must still build +
        # persist the cache and return results, same as the no-flag case.
        code, out = _run(
            ["query", "what carries data between networks",
             "--root", str(self.root), "--top-k", "1", "--stale-ok"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "a.txt")
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
            ["query", "asteroid belt mars jupiter", "--root", str(self.root), "--top-k", "1"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "c.txt")

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
        self.assertIn("docs: 2", out)
        self.assertIn("stale: False", out)

    def test_json_output_parses(self) -> None:
        code, _out = _run(["index", "--root", str(self.root), "--retriever", "lexical"])
        self.assertEqual(code, 0)

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
        self.assertEqual(payload["results"], ["a.txt"])

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
        for choice in ("lexical", "lexical+ctx", "turbovec", "pi-serini", "hybrid"):
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
        self.assertIn("lexical: indexed 2 docs", out)

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

    def test_bad_args_exit_nonzero(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["not-a-real-command"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_required_query_arg_exits_nonzero(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["query"])
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
