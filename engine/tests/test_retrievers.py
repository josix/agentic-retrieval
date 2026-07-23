"""Offline tests for the document-level retriever adapters.

Uses a synthetic in-memory corpus so the lexical retrievers are exercised
without any dataset download.
"""

import unittest

from retrieval.document import Document
from retrieval.retrievers import (
    REGISTRY,
    ContextualLexicalRetriever,
    HybridRetriever,
    LexicalRetriever,
    TreeSitterRetriever,
    build_retriever,
)

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


if __name__ == "__main__":
    unittest.main()
