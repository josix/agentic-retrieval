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
    build_retriever,
)

try:
    import turbovec  # noqa: F401

    _TURBOVEC_INSTALLED = True
except ImportError:
    _TURBOVEC_INSTALLED = False


def _documents():
    return [
        Document("d1", "Routers forward packets between networks and carry data."),
        Document("d2", "Photosynthesis converts sunlight into chemical energy in plants."),
        Document("d3", "The asteroid belt lies between Mars and Jupiter."),
        Document("d4", "Switches direct frames within a local area network domain."),
    ]


class TestLexicalRetriever(unittest.TestCase):
    def test_ranks_relevant_doc_first(self) -> None:
        documents = _documents()
        r = LexicalRetriever()
        r.index(documents)
        self.assertEqual(r.search("what carries data between networks", top_k=1)[0], "d1")
        self.assertEqual(r.search("how do plants convert light into energy", top_k=1)[0], "d2")


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


if __name__ == "__main__":
    unittest.main()
