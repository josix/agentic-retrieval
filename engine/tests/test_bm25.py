"""Tests for BM25Index ordering correctness."""

import pathlib
import sys
import unittest

# Add skill dir to path so retrieval package resolves when run via unittest discover
_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.bm25 import BM25Index  # noqa: E402


class TestBM25Ordering(unittest.TestCase):
    """BM25 should rank the most relevant document highest."""

    def setUp(self) -> None:
        self.docs = [
            "The cat sat on the mat",
            "Dogs and cats are common household pets",
            "Quantum mechanics describes subatomic behaviour",
        ]
        self.index = BM25Index()
        self.index.fit(self.docs)

    def test_relevant_doc_ranks_first(self) -> None:
        results = self.index.query("cat sat mat")
        self.assertGreater(len(results), 0)
        top_idx, top_score = results[0]
        self.assertEqual(top_idx, 0, "First doc should rank highest for 'cat sat mat'")

    def test_second_doc_cat_query(self) -> None:
        results = self.index.query("household pets dogs")
        top_idx, _ = results[0]
        self.assertEqual(top_idx, 1, "Second doc should rank first for pet query")

    def test_scores_sorted_descending(self) -> None:
        results = self.index.query("cat")
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_list(self) -> None:
        results = self.index.query("")
        # All scores are 0, but we still get a list
        self.assertIsInstance(results, list)

    def test_single_doc_corpus(self) -> None:
        idx = BM25Index()
        idx.fit(["only document here"])
        results = idx.query("document")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0)

    def test_unseen_term_zero_contribution(self) -> None:
        results = self.index.query("xyzzy nonexistent term")
        # All scores should be 0 since no term matches
        for _, score in results:
            self.assertAlmostEqual(score, 0.0)

    def test_returns_all_docs(self) -> None:
        results = self.index.query("the")
        self.assertEqual(len(results), len(self.docs))


class TestBM25Tokenizer(unittest.TestCase):
    def test_default_tokenizer_is_plain(self) -> None:
        self.assertEqual(BM25Index().tokenizer, "plain")

    def test_unknown_tokenizer_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            BM25Index(tokenizer="not-a-mode")

    def test_to_dict_from_dict_round_trips_tokenizer(self) -> None:
        idx = BM25Index(tokenizer="code")
        idx.fit(["def getUserById(): pass"])
        data = idx.to_dict()
        self.assertEqual(data["tokenizer"], "code")
        restored = BM25Index.from_dict(data)
        self.assertEqual(restored.tokenizer, "code")
        self.assertEqual(idx.query("user"), restored.query("user"))

    def test_from_dict_defaults_to_plain_when_tokenizer_key_missing(self) -> None:
        idx = BM25Index()
        idx.fit(["hello world"])
        data = idx.to_dict()
        del data["tokenizer"]
        restored = BM25Index.from_dict(data)
        self.assertEqual(restored.tokenizer, "plain")


if __name__ == "__main__":
    unittest.main()
