import asyncio
import builtins
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

from test_cli import _bump_mtime, _write  # noqa: E402

from retrieval import cli, mcp_server, persistence  # noqa: E402
from retrieval.persistence import cached_retrievers, index_dir  # noqa: E402

try:
    import mcp  # noqa: F401

    _MCP_INSTALLED = True
except ImportError:
    _MCP_INSTALLED = False

_ENGINE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._project_tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._project_tmp.name).resolve()
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


class TestResolveRoot(unittest.TestCase):
    def test_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            env = {"RETRIEVAL_ROOT": b, "CLAUDE_PROJECT_DIR": c}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(mcp_server.resolve_server_root(a), pathlib.Path(a).resolve())
                self.assertEqual(mcp_server.resolve_server_root(None), pathlib.Path(b).resolve())
                del os.environ["RETRIEVAL_ROOT"]
                self.assertEqual(mcp_server.resolve_server_root(None), pathlib.Path(c).resolve())
                del os.environ["CLAUDE_PROJECT_DIR"]
                self.assertEqual(mcp_server.resolve_server_root(None), pathlib.Path.cwd().resolve())


class TestSearch(_Base):
    def test_consolidated_lazy_lexical(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers packets")
        self.assertEqual(out["mode"], "consolidated")
        self.assertIn("lexical", out["retrievers"])
        self.assertEqual(out["results"][0]["path"], "a.txt")
        self.assertIn("routers", out["results"][0]["snippet"])
        skipped = {s["name"]: s["reason"] for s in out["skipped"]}
        self.assertIn("turbovec", skipped)
        self.assertIn("not indexed", skipped["turbovec"])
        self.assertIn("lexical", cached_retrievers(self.root))

    def test_no_snippet_and_single_mode(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers", include_snippet=False)
        self.assertIsNone(out["results"][0]["snippet"])
        single = mcp_server.search_domain_context(self.root, "routers", retriever="lexical")
        self.assertEqual(single["mode"], "single")
        self.assertEqual(single["results"][0]["path"], "a.txt")
        self.assertIn("docid", single["results"][0])

    def test_unindexed_heavy_single_retriever_is_skipped_not_raised(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers", retriever="turbovec")
        self.assertEqual(out["results"], [])
        self.assertEqual(out["skipped"][0]["name"], "turbovec")
        self.assertIn("index_project", out["skipped"][0]["reason"])
        self.assertIn("stale", out["index"])

    def test_index_block_and_truncated_flag(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers packets")
        self.assertFalse(out["index"]["stale"])
        self.assertTrue(out["index"]["created_at"])
        self.assertFalse(out["results"][0]["truncated"])

    def test_top_k_clamped(self) -> None:
        for i in range(30):
            _write(self.root / f"d{i}.txt", f"routers forward packets variant {i}")
        out = mcp_server.search_domain_context(self.root, "routers packets", top_k=50,
                                               retriever="lexical")
        self.assertLessEqual(len(out["results"]), mcp_server._MAX_TOP_K)

    def test_snippet_byte_cap_sets_truncated(self) -> None:
        _write(self.root / "big.txt", "\n".join("bigword " * 20 for _ in range(100)))
        rows = [{"path": "big.txt", "start_line": 1, "end_line": 100}]
        mcp_server._attach_snippets(self.root, rows, True, 200)
        self.assertLessEqual(len(rows[0]["snippet"].encode()), mcp_server._MAX_SNIPPET_BYTES)
        self.assertTrue(rows[0]["truncated"])

    def test_guarded_returns_error_dict(self) -> None:
        for args in (("  ",), ("x", "nope")):
            out = mcp_server._guarded(mcp_server.search_domain_context, self.root, *args)
            self.assertIn("error", out)
            self.assertNotIn("results", out)

    def test_validation(self) -> None:
        for kwargs in ({"query": "  "}, {"query": "x", "top_k": 0},
                       {"query": "x", "weights": {"bogus": 1}}, {"query": "x", "pool": 0}):
            with self.assertRaises(ValueError):
                mcp_server.search_domain_context(self.root, **kwargs)

    def test_weights_serialized(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers", weights={"lexical": 2})
        self.assertIn("lexical", out["retrievers"])

    def test_stale_auto_refreshes_lexical(self) -> None:
        mcp_server.search_domain_context(self.root, "routers")
        _write(self.root / "a.txt", "routers forward packets; brand new zebra content")
        _bump_mtime(self.root / "a.txt")
        meta = cli.read_meta(self.root, "lexical")
        self.assertTrue(cli.is_stale(self.root, meta))
        out = mcp_server.search_domain_context(self.root, "zebra")
        self.assertEqual(out["results"][0]["path"], "a.txt")
        self.assertFalse(cli.is_stale(self.root, cli.read_meta(self.root, "lexical")))

    def test_stale_ok_true_reports_stale_in_both_modes(self) -> None:
        mcp_server.search_domain_context(self.root, "routers")
        _write(self.root / "a.txt", "routers forward packets; brand new zebra content")
        _bump_mtime(self.root / "a.txt")
        for retriever in ("all", "lexical"):
            out = mcp_server.search_domain_context(
                self.root, "routers", retriever=retriever, stale_ok=True)
            self.assertEqual(out["stale"], ["lexical"])
        self.assertTrue(cli.is_stale(self.root, cli.read_meta(self.root, "lexical")))


class TestStatusAndIndex(_Base):
    def test_status_then_index(self) -> None:
        before = mcp_server.index_status(self.root)
        lexical = next(s for s in before["strategies"] if s["name"] == "lexical")
        self.assertFalse(lexical["indexed"])
        self.assertIn("index_project", before["recommendation"])

        outer = io.StringIO()
        with redirect_stdout(outer):
            res = mcp_server.index_project(self.root, "lexical")
        self.assertEqual(outer.getvalue(), "")
        self.assertTrue(res["ok"])
        self.assertEqual(res["exit_code"], 0)
        self.assertTrue(res["log"])
        self.assertIn("lexical", res["status"]["ready"])

    def test_bad_retriever(self) -> None:
        with self.assertRaises(ValueError):
            mcp_server.index_project(self.root, "nope")


class TestSnippet(_Base):
    def test_unsafe_paths(self) -> None:
        for rel in ("doc.pdf", "../escape.txt"):
            snippet, note = mcp_server._read_snippet(self.root, rel, 1, 2, 5)
            self.assertIsNone(snippet)
            self.assertTrue(note)

    def test_line_window(self) -> None:
        _write(self.root / "m.txt", "l1\nl2\nl3\nl4\n")
        snippet, note = mcp_server._read_snippet(self.root, "m.txt", 2, 4, 2)
        self.assertEqual(snippet, "l2\nl3")
        self.assertIsNone(note)


class TestCliEntry(_Base):
    def test_missing_extra_exits_one(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError("no mcp")
            return real_import(name, *args, **kwargs)

        err = io.StringIO()
        with mock.patch("builtins.__import__", fake_import), redirect_stderr(err):
            code = cli.main(["mcp", "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("--extra mcp", err.getvalue())

    def test_protect_stdout_separates_streams(self) -> None:
        code = (
            "import os, sys\n"
            "from retrieval import mcp_server\n"
            "mcp_server._protect_stdout()\n"
            "os.write(1, b'NATIVE')\n"
            "sys.stdout.write('PROTO'); sys.stdout.flush()\n"
        )
        env = {**os.environ, "PYTHONPATH": _ENGINE_DIR}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env=env, timeout=60)
        self.assertEqual(proc.stdout, "PROTO")
        self.assertIn("NATIVE", proc.stderr)


@unittest.skipUnless(_MCP_INSTALLED, "requires the 'mcp' extra")
class TestServer(_Base):
    def test_tools_and_call(self) -> None:
        server = mcp_server.build_server(self.root)
        tools = asyncio.run(server.list_tools())
        self.assertEqual(
            {t.name for t in tools},
            {"search_domain_context", "index_project", "index_status"},
        )
        result = asyncio.run(server.call_tool("search_domain_context", {"query": "photosynthesis"}))
        self.assertIn("b.txt", str(result))

    def test_registration_names(self) -> None:
        server = mcp_server.build_server(self.root)
        tools = {t.name: t for t in asyncio.run(server.list_tools())}
        self.assertEqual(set(tools), {"search_domain_context", "index_project", "index_status"})
        self.assertFalse(tools["search_domain_context"].annotations.readOnlyHint)
        self.assertTrue(tools["index_status"].annotations.readOnlyHint)
        self.assertIn("USE THIS FIRST", tools["search_domain_context"].description)
        prompts = asyncio.run(server.list_prompts())
        self.assertEqual({p.name for p in prompts}, {"ground_implementation"})
        resources = asyncio.run(server.list_resources())
        self.assertEqual({str(r.uri) for r in resources}, {mcp_server.STATUS_RESOURCE_URI})
        self.assertIn("search_domain_context", server.instructions)

    def test_prompt_and_resource_content(self) -> None:
        server = mcp_server.build_server(self.root)
        got = asyncio.run(server.get_prompt("ground_implementation", {"task": "add retry"}))
        text = got.messages[0].content.text
        self.assertIn("search_domain_context", text)
        self.assertIn("add retry", text)
        res = asyncio.run(server.read_resource(mcp_server.STATUS_RESOURCE_URI))
        self.assertIn("strategies", list(res)[0].content)

    def test_stdio_round_trip(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "retrieval", "mcp", "--root", str(self.root)],
            env={**os.environ, "RETRIEVAL_INDEX_DIR": self._cache_tmp.name,
                 "PYTHONPATH": _ENGINE_DIR},
        )

        async def go():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool("index_status", {})

        result = asyncio.run(asyncio.wait_for(go(), timeout=60))
        self.assertFalse(result.isError)


def _mark_slot_ctx(root: pathlib.Path) -> None:
    """Relabel the shared lexical slot as a (paid) lexical+ctx build."""
    meta_path = index_dir(root) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["retriever_name"] = "lexical+ctx"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _make_stale(root: pathlib.Path) -> None:
    _write(root / "a.txt", "routers forward packets; brand new zebra content")
    _bump_mtime(root / "a.txt")


class TestLexicalCtxSafety(_Base):
    def setUp(self) -> None:
        super().setUp()
        mcp_server.search_domain_context(self.root, "routers")  # builds lexical
        _mark_slot_ctx(self.root)

    def test_stale_ctx_slot_never_rebuilt(self) -> None:
        before = (index_dir(self.root) / "meta.json").read_text(encoding="utf-8")
        _make_stale(self.root)
        with mock.patch.object(cli, "_build_and_save", side_effect=AssertionError("built")):
            for retriever in ("all", "lexical"):
                for stale_ok in (None, False):
                    out = mcp_server.search_domain_context(
                        self.root, "routers", retriever=retriever, stale_ok=stale_ok)
                    self.assertEqual(out["stale"], ["lexical"])
                    self.assertIn("LLM", out["hint"])
        self.assertEqual((index_dir(self.root) / "meta.json").read_text(encoding="utf-8"), before)

    def test_search_existing_ctx_slot(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers", retriever="lexical+ctx")
        self.assertNotIn("error", out)
        self.assertEqual(out["retrievers"], ["lexical+ctx"])
        _make_stale(self.root)
        out = mcp_server.search_domain_context(
            self.root, "routers", retriever="lexical+ctx", stale_ok=True)
        self.assertEqual(out["results"][0]["path"], "a.txt")
        out = mcp_server.search_domain_context(
            self.root, "routers", retriever="lexical+ctx", stale_ok=False)
        self.assertEqual(out["error_type"], "OptInRequired")

    def test_ctx_search_without_index_refused(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            _write(pathlib.Path(other) / "x.txt", "hello")
            out = mcp_server.search_domain_context(
                pathlib.Path(other).resolve(), "hello", retriever="lexical+ctx")
            self.assertIn("error", out)
            self.assertIn("retrieval index --retriever lexical+ctx", out["hint"])

    def test_index_project_refuses_ctx_and_overwrite(self) -> None:
        with mock.patch.object(cli, "_cmd_index", side_effect=AssertionError("indexed")):
            for retriever in ("lexical+ctx", "lexical", "all"):
                out = mcp_server.index_project(self.root, retriever)
                self.assertEqual(out["error_type"], "OptInRequired", retriever)
                self.assertIn("hint", out)

    def test_status_recommends_cli_for_stale_ctx(self) -> None:
        _make_stale(self.root)
        status = mcp_server.index_status(self.root)
        self.assertIn("retrieval index --retriever lexical+ctx", status["recommendation"])


class TestHints(_Base):
    def test_search_did_not_run_is_not_no_precedent(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers", retriever="turbovec")
        self.assertIn("did not run", out["hint"])
        self.assertNotIn("likely has no precedent", out["hint"])
        self.assertNotIn("Only lexical", out["hint"])
        self.assertIn("retriever='lexical'", out["hint"])

    def test_lexical_failure_in_consolidated_mode(self) -> None:
        with mock.patch.object(cli, "_load_or_rebuild", side_effect=RuntimeError("boom")):
            out = mcp_server.search_domain_context(self.root, "routers", stale_ok=True)
        self.assertEqual(out["retrievers"], [])
        self.assertIn("did not run", out["hint"])
        self.assertIn("boom", out["hint"])

    def test_ran_and_found_nothing_is_no_precedent(self) -> None:
        hint = mcp_server._hint([], [], [], used=["lexical"])
        self.assertIn("likely has no precedent", hint)
        self.assertNotIn("did not run", hint)

    def test_consolidated_skipped_hint_names_what_was_searched(self) -> None:
        out = mcp_server.search_domain_context(self.root, "routers")
        self.assertIn("Searched only: lexical", out["hint"])


class TestRefreshPolicy(_Base):
    def test_only_refreshes_when_lexical_is_searched(self) -> None:
        with mock.patch.object(mcp_server, "_refresh_lexical_if_stale") as refresh:
            self.assertTrue(mcp_server._resolve_stale_ok(self.root, None, "turbovec", {}))
            refresh.assert_not_called()
            for retriever in ("all", "lexical"):
                mcp_server._resolve_stale_ok(self.root, None, retriever, {})
            self.assertEqual(refresh.call_count, 2)
            mcp_server._resolve_stale_ok(self.root, True, "lexical", {})
            self.assertEqual(refresh.call_count, 2)


class TestConcurrency(_Base):
    def test_status_and_search_not_blocked_by_build(self) -> None:
        mcp_server.search_domain_context(self.root, "routers")
        with mock.patch.object(mcp_server, "_BUILD_LOCK", threading.Lock()) as lock:
            lock.acquire()
            status = mcp_server.index_status(self.root)
            self.assertIn("lexical", status["ready"])
            out = mcp_server.search_domain_context(self.root, "routers")
            self.assertTrue(out["build_in_progress"])
            self.assertEqual(out["results"][0]["path"], "a.txt")
            self.assertIn("build is in progress", out["hint"])
            second = mcp_server.index_project(self.root, "lexical")
            self.assertEqual(second["error_type"], "BuildInProgress")

    def test_build_in_progress_never_writes(self) -> None:
        with mock.patch.object(mcp_server, "_BUILD_LOCK", threading.Lock()) as lock:
            lock.acquire()
            with mock.patch.object(cli, "_build_and_save", side_effect=AssertionError("built")):
                out = mcp_server.search_domain_context(self.root, "routers", build_missing=True)
        self.assertEqual(out["retrievers"], [])
        self.assertIn("did not run", out["hint"])

    def test_pi_serini_skipped_during_build(self) -> None:
        with mock.patch.object(mcp_server, "_BUILD_LOCK", threading.Lock()) as lock:
            lock.acquire()
            single = mcp_server.search_domain_context(self.root, "routers", retriever="pi-serini")
            self.assertEqual(single["results"], [])
            self.assertEqual(single["skipped"][0]["name"], "pi-serini")
            names, _ = mcp_server._available_names(self.root)
            with mock.patch.object(mcp_server, "_available_names",
                                   return_value=(list(names) + ["pi-serini"], [])):
                out = mcp_server.search_domain_context(self.root, "routers")
        self.assertNotIn("pi-serini", out["retrievers"])
        self.assertIn("pi-serini", [s["name"] for s in out["skipped"]])

    def test_lock_released_after_build(self) -> None:
        mcp_server.index_project(self.root, "lexical")
        self.assertFalse(mcp_server._BUILD_LOCK.locked())

    def test_stdout_restored_after_overlapping_handlers(self) -> None:
        before = (sys.stdout, sys.stderr)
        with mcp_server._diverted(), mcp_server._diverted():
            print("stray")  # goes to real stderr, not stdout
        self.assertEqual((sys.stdout, sys.stderr), before)


class TestPerformance(_Base):
    def test_retriever_reused_and_evicted_on_rebuild(self) -> None:
        real = mcp_server.load_index
        with mock.patch.object(mcp_server, "load_index", side_effect=real) as load:
            mcp_server.search_domain_context(self.root, "routers", retriever="lexical")
            mcp_server.search_domain_context(self.root, "routers", retriever="lexical")
            mcp_server.search_domain_context(self.root, "packets", retriever="lexical")
            self.assertEqual(load.call_count, 1)
            _make_stale(self.root)
            out = mcp_server.search_domain_context(self.root, "zebra", retriever="lexical")
            self.assertEqual(out["results"][0]["path"], "a.txt")
            mcp_server.search_domain_context(self.root, "zebra", retriever="lexical")
            self.assertEqual(load.call_count, 2)

    def test_one_fingerprint_walk_per_request(self) -> None:
        mcp_server.search_domain_context(self.root, "routers")
        real = persistence.compute_fingerprint
        with mock.patch.object(persistence, "compute_fingerprint", side_effect=real) as fp:
            mcp_server.search_domain_context(self.root, "routers")
            self.assertEqual(fp.call_count, 1)
            fp.reset_mock()
            mcp_server.search_domain_context(self.root, "routers", retriever="lexical")
            self.assertEqual(fp.call_count, 1)

    def test_snippet_reads_only_window_and_once_per_file(self) -> None:
        big = self.root / "big.txt"
        _write(big, "".join(f"line {i}\n" for i in range(5000)))
        files = mcp_server._SnippetFiles()
        first = mcp_server._read_snippet(self.root, "big.txt", 2, 3, 5, files)
        again = mcp_server._read_snippet(self.root, "big.txt", 1, 2, 5, files)
        self.assertEqual(first, ("line 1\nline 2", None))
        self.assertEqual(again, ("line 0\nline 1", None))
        self.assertLessEqual(len(files._lines[big.resolve()][0]), 5)
        with mock.patch.object(pathlib.Path, "open", side_effect=AssertionError("reopened")):
            mcp_server._read_snippet(self.root, "big.txt", 1, 3, 5, files)


class TestPluginManifest(unittest.TestCase):
    def test_mcp_server_declared_in_plugin_json_only(self) -> None:
        repo = pathlib.Path(_ENGINE_DIR).parent
        self.assertFalse((repo / ".mcp.json").exists())
        plugin = json.loads((repo / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
        server = plugin["mcpServers"]["retrieval"]
        self.assertEqual(server["command"], "sh")
        self.assertEqual(server["args"], ["${CLAUDE_PLUGIN_ROOT}/scripts/retrieval-mcp.sh"])


if __name__ == "__main__":
    unittest.main()
