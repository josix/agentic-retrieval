"""Offline tests for the pluggable LLM contextualizer wiring.

No network or anthropic dependency: a stub contextualizer stands in for the
LLM, verifying that ContextualRetriever.build() routes through any
``(chunk, doc_chunks) -> str`` strategy and that document reconstruction is
deterministic.
"""

import unittest

from retrieval.chunker import Chunk
from retrieval.index import ContextualRetriever
from retrieval.llm_contextualizer import LLMContextualizer


def _chunk(doc_id: str, text: str, position: int, heading: str = "H") -> Chunk:
    return Chunk(
        id=f"{doc_id}_{position}",
        doc_id=doc_id,
        doc_title="Doc",
        heading=heading,
        text=text,
        position=position,
    )


class TestPluggableContextualizer(unittest.TestCase):
    def test_build_routes_through_custom_contextualizer(self) -> None:
        chunks = [
            _chunk("d", "alpha packets", 0),
            _chunk("d", "beta frames", 1),
        ]
        seen = []

        def stub(chunk, doc_chunks):
            seen.append((chunk.position, len(doc_chunks)))
            return f"SENTINEL_{chunk.position} {chunk.text}"

        retriever = ContextualRetriever()
        retriever.build(chunks, contextualizer=stub)

        # The stub saw every chunk, each with its full document context.
        self.assertEqual(sorted(seen), [(0, 2), (1, 2)])
        # The sentinel token only exists in the contextualized index, so a
        # query for it must retrieve the matching chunk.
        results = retriever.search("SENTINEL_1", top_k=1)
        self.assertEqual(results[0].chunk.position, 1)

    def test_custom_contextualizer_implies_context(self) -> None:
        chunks = [_chunk("d", "body", 0)]
        retriever = ContextualRetriever()
        # use_context defaults False, but passing a contextualizer must engage it.
        retriever.build(chunks, contextualizer=lambda c, d: f"CTX {c.text}")
        self.assertIn("CTX", retriever._texts[0])


class TestDocumentReconstruction(unittest.TestCase):
    def test_chunks_joined_in_position_order(self) -> None:
        out_of_order = [
            _chunk("d", "third", 2),
            _chunk("d", "first", 0),
            _chunk("d", "second", 1),
        ]
        doc = LLMContextualizer._document_text(out_of_order)
        self.assertEqual(doc, "first\n\nsecond\n\nthird")


if __name__ == "__main__":
    unittest.main()
