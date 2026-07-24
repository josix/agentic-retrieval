"""Tests for TfidfIndex cosine correctness and ordering."""

import pathlib
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.tfidf import TfidfIndex, tokenize  # noqa: E402


class TestTokenize(unittest.TestCase):
    def test_lowercase(self) -> None:
        tokens = tokenize("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_strips_punctuation(self) -> None:
        tokens = tokenize("cat, dog! fish?")
        self.assertIn("cat", tokens)
        self.assertIn("dog", tokens)
        self.assertIn("fish", tokens)
        self.assertNotIn(",", tokens)

    def test_empty_string(self) -> None:
        self.assertEqual(tokenize(""), [])

    def test_numbers_included(self) -> None:
        tokens = tokenize("layer 2 switching")
        self.assertIn("2", tokens)


class TestTokenizeCodeMode(unittest.TestCase):
    def test_plain_mode_byte_identical_to_no_arg(self) -> None:
        text = "getUserById HTTPSConnection load_chunk_documents"
        self.assertEqual(tokenize(text), tokenize(text, "plain"))

    def test_code_mode_snake_case_contains_whole_and_subtokens(self) -> None:
        tokens = tokenize("load_chunk_documents", "code")
        self.assertIn("load_chunk_documents", tokens)
        self.assertIn("load", tokens)
        self.assertIn("chunk", tokens)
        self.assertIn("documents", tokens)

    def test_code_mode_camel_case_splits(self) -> None:
        tokens = tokenize("getUserById", "code")
        self.assertIn("user", tokens)
        self.assertIn("by", tokens)
        self.assertIn("id", tokens)

    def test_code_mode_acronym_boundary(self) -> None:
        tokens = tokenize("HTTPSConnection", "code")
        self.assertIn("https", tokens)
        self.assertIn("connection", tokens)

    def test_code_mode_non_splitting_token_not_duplicated(self) -> None:
        self.assertEqual(tokenize("chunk", "code"), ["chunk"])

    def test_unknown_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            tokenize("hello", "not-a-mode")


class TestTfidfOrdering(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            "photosynthesis converts light into chemical energy in plants",
            "the solar system orbits the sun with eight planets",
            "routing forwards packets between network segments efficiently",
        ]
        self.index = TfidfIndex()
        self.index.fit(self.docs)

    def test_scores_sorted_descending(self) -> None:
        results = self.index.query("photosynthesis light energy")
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_relevant_doc_ranks_first(self) -> None:
        results = self.index.query("photosynthesis plants")
        top_idx, top_score = results[0]
        self.assertEqual(top_idx, 0)

    def test_network_doc_ranks_first_for_routing(self) -> None:
        results = self.index.query("routing packets network")
        top_idx, _ = results[0]
        self.assertEqual(top_idx, 2)

    def test_scores_are_non_negative(self) -> None:
        results = self.index.query("solar system")
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)

    def test_empty_corpus_returns_empty(self) -> None:
        idx = TfidfIndex()
        idx.fit([])
        results = idx.query("anything")
        self.assertEqual(results, [])

    def test_single_doc_scores_positive(self) -> None:
        idx = TfidfIndex()
        idx.fit(["hello world"])
        results = idx.query("hello")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0.0)

    def test_returns_all_docs(self) -> None:
        results = self.index.query("the")
        self.assertEqual(len(results), len(self.docs))


class TestTfidfTokenizer(unittest.TestCase):
    def test_default_tokenizer_is_plain(self) -> None:
        self.assertEqual(TfidfIndex().tokenizer, "plain")

    def test_unknown_tokenizer_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            TfidfIndex(tokenizer="not-a-mode")

    def test_to_dict_from_dict_round_trips_tokenizer(self) -> None:
        idx = TfidfIndex(tokenizer="code")
        idx.fit(["def getUserById(): pass"])
        data = idx.to_dict()
        self.assertEqual(data["tokenizer"], "code")
        restored = TfidfIndex.from_dict(data)
        self.assertEqual(restored.tokenizer, "code")
        self.assertEqual(idx.query("user"), restored.query("user"))

    def test_from_dict_defaults_to_plain_when_tokenizer_key_missing(self) -> None:
        idx = TfidfIndex()
        idx.fit(["hello world"])
        data = idx.to_dict()
        del data["tokenizer"]
        restored = TfidfIndex.from_dict(data)
        self.assertEqual(restored.tokenizer, "plain")


if __name__ == "__main__":
    unittest.main()
