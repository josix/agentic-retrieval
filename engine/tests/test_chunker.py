"""Tests for retrieval.chunker line-span tracking.

Covers span correctness on a realistic multi-heading document plus the
documented edge cases: blank text, no trailing newline, the no-heading
fallback chunk, heading-line inclusion at a chunk boundary, and a
multi-paragraph chunk that spans a blank-line gap.
"""

import pathlib
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.chunker import chunk_document  # noqa: E402


class TestChunkDocumentSpans(unittest.TestCase):
    """Span correctness against the computer_networks.txt fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_path = (
            pathlib.Path(__file__).parent / "fixtures" / "computer_networks.txt"
        )
        cls.text = cls.fixture_path.read_text(encoding="utf-8")
        cls.chunks = chunk_document("computer_networks", cls.text)

    def test_every_chunk_has_a_positive_span(self) -> None:
        for chunk in self.chunks:
            self.assertGreaterEqual(chunk.start_line, 1)
            self.assertGreaterEqual(chunk.end_line, chunk.start_line)

    def test_title_chunk_spans_the_first_line(self) -> None:
        title_chunk = self.chunks[0]
        self.assertEqual(title_chunk.start_line, 1)
        self.assertEqual(title_chunk.end_line, 1)

    def test_heading_chunk_span_extends_to_include_heading_line(self) -> None:
        routing = next(c for c in self.chunks if c.heading == "Routing")
        # "Routing:" is line 6 in the fixture, its paragraph is line 7.
        self.assertEqual(routing.start_line, 6)
        self.assertEqual(routing.end_line, 7)

    def test_spans_are_non_overlapping_and_increasing(self) -> None:
        prev_end = 0
        for chunk in self.chunks:
            self.assertGreater(chunk.start_line, prev_end - 1)
            prev_end = chunk.end_line


class TestChunkDocumentEdgeCases(unittest.TestCase):
    def test_blank_text_returns_empty_list(self) -> None:
        self.assertEqual(chunk_document("d", ""), [])
        self.assertEqual(chunk_document("d", "   \n\n  \n"), [])

    def test_no_trailing_newline_single_line(self) -> None:
        chunks = chunk_document("d", "just one line, no trailing newline")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 1)

    def test_fallback_chunk_spans_whole_document(self) -> None:
        # Every line looks like a heading, so no body paragraph is ever
        # accumulated -- chunk_document falls back to a single whole-text
        # chunk spanning every source line (1..len(lines)).
        text = "# H1\n## H2\n### H3\n"
        chunks = chunk_document("d", text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 3)

    def test_no_heading_multiline_body_is_a_single_normal_chunk(self) -> None:
        # No headings and no blank-line breaks -> one paragraph via the
        # normal (non-fallback) path, spanning every line.
        text = "line one\nline two\nline three"
        chunks = chunk_document("d", text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 3)

    def test_heading_line_included_in_chunk_span(self) -> None:
        text = "Title\n\nOverview:\nbody text here\n"
        chunks = chunk_document("d", text)
        overview = next(c for c in chunks if c.heading == "Overview")
        # "Overview:" is line 3, its paragraph is line 4.
        self.assertEqual(overview.start_line, 3)
        self.assertEqual(overview.end_line, 4)

    def test_multi_paragraph_chunk_spans_blank_line_gap(self) -> None:
        text = "# Title\n\n## Sec\npara one\n\npara two\n"
        chunks = chunk_document("d", text, target_chars=1000)
        sec = next(c for c in chunks if c.heading == "Sec")
        # "## Sec" is line 3; "para one" is line 4; "para two" is line 6
        # (line 5 is the blank separator) -- span must cover the gap.
        self.assertEqual(sec.start_line, 3)
        self.assertEqual(sec.end_line, 6)
        self.assertIn("para one", sec.text)
        self.assertIn("para two", sec.text)

    def test_size_exceeded_split_does_not_reinclude_heading_line(self) -> None:
        # Two paragraphs under the same heading, forced into separate
        # chunks by a tiny target_chars -- only the first chunk should
        # have its span extended to the heading line.
        text = "Title\n\nSection:\nfirst paragraph text\n\nsecond paragraph text\n"
        chunks = chunk_document("d", text, target_chars=5)
        section_chunks = [c for c in chunks if c.heading == "Section"]
        self.assertEqual(len(section_chunks), 2)
        first, second = section_chunks
        # "Section:" is line 3; first paragraph is line 4.
        self.assertEqual(first.start_line, 3)
        self.assertEqual(first.end_line, 4)
        # second paragraph is line 6; must not reach back to the heading line.
        self.assertEqual(second.start_line, 6)
        self.assertEqual(second.end_line, 6)


if __name__ == "__main__":
    unittest.main()
