"""Tests for retrieval.ast_chunker.

Splits into an always-on group (suffix -> language mapping, which needs no
optional extras) and a group gated on the ``treesitter`` extra being
installed (actual AST-boundary chunking).
"""

import unittest
from pathlib import Path

from retrieval.ast_chunker import _process_siblings, chunk_code, language_for_path

try:
    import tree_sitter_language_pack  # noqa: F401

    _TREESITTER_INSTALLED = True
except ImportError:
    _TREESITTER_INSTALLED = False


class _FakePoint:
    def __init__(self, row: int) -> None:
        self.row = row


class _FakeNameNode:
    def __init__(self, name: str) -> None:
        self.text = name.encode("utf-8")


class _FakeNode:
    """Minimal stand-in for a tree-sitter node: enough attributes for
    ``_process_siblings``/``_hard_split_leaf`` to treat it as a true AST
    leaf (no named children) whose own type is a named scope."""

    def __init__(self, node_type: str, name: str, text: bytes) -> None:
        self.type = node_type
        self.children: list = []
        self.is_named = True
        self.start_byte = 0
        self.end_byte = len(text)
        self.start_point = _FakePoint(0)
        self.end_point = _FakePoint(text.count(b"\n"))
        self._name = name

    def child_by_field_name(self, field: str):
        return _FakeNameNode(self._name) if field == "name" else None


class TestLanguageForPath(unittest.TestCase):
    """Suffix -> language mapping needs no optional extras."""

    def test_python_suffix_maps_to_python(self) -> None:
        self.assertEqual(language_for_path("app.py"), "python")

    def test_unmapped_suffix_returns_none(self) -> None:
        self.assertIsNone(language_for_path("README.md"))
        self.assertIsNone(language_for_path("notes.txt"))

    def test_case_insensitive_suffix_match(self) -> None:
        self.assertEqual(language_for_path("APP.PY"), "python")


@unittest.skipUnless(_TREESITTER_INSTALLED, "tree-sitter-language-pack not installed")
class TestChunkCodeAstBoundaries(unittest.TestCase):
    """Actual AST-boundary chunking; only runs when the extra is installed."""

    def test_two_functions_and_a_class_align_to_def_spans(self) -> None:
        src = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
            "\n"
            "\n"
            "class Bar:\n"
            "    def baz(self):\n"
            "        return 3\n"
        )
        chunks = chunk_code("x.py", src, "python", max_chars=1200)
        # Small enough to fit as one merged top-level chunk.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 11)
        self.assertEqual(chunks[0].context, "")

    def test_breadcrumb_reflects_enclosing_scope_when_split(self) -> None:
        src = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
            "\n"
            "\n"
            "class Bar:\n"
            "    def baz(self):\n"
            "        return 3\n"
        )
        # A tiny budget forces recursion into every scope, so nested chunks
        # carry a "Class.method" (or "Class") breadcrumb.
        chunks = chunk_code("x.py", src, "python", max_chars=5)
        contexts = {c.context for c in chunks}
        self.assertIn("foo", contexts)
        self.assertIn("bar", contexts)
        self.assertIn("Bar", contexts)
        self.assertIn("Bar.baz", contexts)
        # Spans stay 1-based and non-decreasing.
        for chunk in chunks:
            self.assertGreaterEqual(chunk.start_line, 1)
            self.assertLessEqual(chunk.start_line, chunk.end_line)

    def test_unicode_line_numbers_are_correct(self) -> None:
        src = "# 日本語コメント\ndef foo():\n    return \"café\"\n"
        chunks = chunk_code("u.py", src, "python", max_chars=1200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 3)

    def test_partial_syntax_error_is_tolerated(self) -> None:
        # First def is malformed; the second, valid def should still chunk.
        src = "def foo(:\n    pass\n\ndef bar():\n    return 1\n"
        chunks = chunk_code("s.py", src, "python", max_chars=1200)
        self.assertGreater(len(chunks), 0)

    def test_severely_broken_source_returns_empty(self) -> None:
        src = "@#$%^&*(!!!\n???***"
        self.assertEqual(chunk_code("b.py", src, "python"), [])

    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(chunk_code("e.py", "", "python"), [])
        self.assertEqual(chunk_code("e.py", "   \n  \n", "python"), [])

    def test_unsupported_grammar_returns_empty_for_fallback(self) -> None:
        self.assertEqual(chunk_code("x.foo", "some text", "not-a-real-language"), [])

    def test_sequential_real_files_do_not_segfault(self) -> None:
        """Regression guard: tree-sitter==0.26.0 has a native
        memory-corruption bug that segfaults (SIGSEGV) after a handful of
        sequential ``parser.parse()`` calls in one process (see the comment
        in ``chunk_code``, and the ``treesitter`` extra's version pin in
        pyproject.toml). A segfault kills the whole test process rather
        than raising a catchable exception, so this test can't assert
        cleanly on failure — its value is as a smoke guard: if the pin is
        ever loosened and the regression resurfaces, this test (run across
        every real source file in the package, several times over) is the
        one most likely to crash the suite and point back here.
        """
        package_dir = Path(__file__).resolve().parent.parent / "retrieval"
        files = sorted(package_dir.glob("*.py"))
        self.assertGreater(len(files), 3, "expected several real source files to chunk")
        for _ in range(3):  # repeat to raise the odds of tripping the bug
            for path in files:
                chunks = chunk_code(str(path), path.read_text(), "python")
                self.assertIsInstance(chunks, list)


@unittest.skipIf(_TREESITTER_INSTALLED, "tree-sitter-language-pack is installed")
class TestChunkCodeGracefulDegradation(unittest.TestCase):
    """When the treesitter extra isn't installed, chunking must fail loudly."""

    def test_raises_runtime_error_when_uninstalled(self) -> None:
        with self.assertRaises(RuntimeError):
            chunk_code("x.py", "def foo():\n    pass\n", "python")


class TestOversizedLeafHardSplitBreadcrumb(unittest.TestCase):
    """Regression: an oversized node that is itself a named scope but has no
    named children (a true AST leaf) must keep its own name in the
    hard-split chunks' breadcrumb, not lose it.

    Runs unconditionally (no ``treesitter`` extra needed): it drives
    ``_process_siblings`` directly with a synthetic node so the "own type is
    in scope_types" branch is exercised without depending on a real
    grammar producing a childless scope node.
    """

    def test_hard_split_leaf_keeps_own_scope_name_in_context(self) -> None:
        raw_bytes = b"x = 1\n" * 400  # oversized relative to max_chars below
        node = _FakeNode("function_definition", "bigfunc", raw_bytes)
        chunks: list = []
        _process_siblings(
            [node], raw_bytes, max_chars=50,
            scope_types={"function_definition"}, ancestors=[], chunks=chunks,
        )
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk.context, "bigfunc")


if __name__ == "__main__":
    unittest.main()
