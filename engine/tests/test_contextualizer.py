"""Tests for contextualizer determinism and content correctness."""

import pathlib
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.chunker import Chunk  # noqa: E402
from retrieval.contextualizer import contextualize  # noqa: E402


def _make_chunk(
    doc_id: str = "test_doc",
    doc_title: str = "Test Document",
    heading: str = "Section One",
    text: str = "This is the chunk text.",
    position: int = 0,
) -> Chunk:
    return Chunk(
        id=f"{doc_id}_{position}",
        doc_id=doc_id,
        doc_title=doc_title,
        heading=heading,
        text=text,
        position=position,
        start_line=1,
        end_line=1,
    )


class TestContextualizerDeterminism(unittest.TestCase):
    """contextualize() must be byte-identical on repeated calls."""

    def test_same_input_same_output(self) -> None:
        chunk = _make_chunk()
        all_chunks = [chunk]
        result1 = contextualize(chunk, all_chunks)
        result2 = contextualize(chunk, all_chunks)
        self.assertEqual(result1, result2)

    def test_deterministic_with_neighbor(self) -> None:
        prev = _make_chunk(text="Previous chunk content. More text.", position=0)
        curr = _make_chunk(text="Current chunk content.", position=1)
        all_chunks = [prev, curr]
        r1 = contextualize(curr, all_chunks)
        r2 = contextualize(curr, all_chunks)
        self.assertEqual(r1, r2)


class TestContextualizerContent(unittest.TestCase):
    """Contextualized text must include title and heading."""

    def test_contains_title(self) -> None:
        chunk = _make_chunk(doc_title="Computer Networks", heading="Routing")
        result = contextualize(chunk, [chunk])
        self.assertIn("Computer Networks", result)

    def test_contains_heading(self) -> None:
        chunk = _make_chunk(doc_title="Computer Networks", heading="Routing")
        result = contextualize(chunk, [chunk])
        self.assertIn("Routing", result)

    def test_contains_chunk_text(self) -> None:
        chunk = _make_chunk(text="It forwards packets between segments.")
        result = contextualize(chunk, [chunk])
        self.assertIn("It forwards packets between segments.", result)

    def test_no_heading_degrades_gracefully(self) -> None:
        chunk = _make_chunk(heading="", doc_title="Solar System")
        result = contextualize(chunk, [chunk])
        self.assertIn("Solar System", result)
        self.assertIn(chunk.text, result)

    def test_breadcrumb_format(self) -> None:
        chunk = _make_chunk(doc_title="Photosynthesis", heading="Calvin Cycle")
        result = contextualize(chunk, [chunk])
        self.assertIn("[Photosynthesis > Calvin Cycle]", result)

    def test_previous_chunk_sentence_included(self) -> None:
        prev = _make_chunk(
            text="Chlorophyll absorbs photons primarily in the red and blue regions.",
            position=0,
        )
        curr = _make_chunk(text="The Calvin cycle fixes carbon.", position=1)
        all_chunks = [prev, curr]
        result = contextualize(curr, all_chunks)
        # Should include at least the start of the previous sentence
        self.assertIn("Chlorophyll", result)


if __name__ == "__main__":
    unittest.main()
