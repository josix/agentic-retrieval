"""Tests for TfidfIndex cosine correctness and ordering."""

import pathlib
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.tfidf import TOKENIZER_MODES, TfidfIndex, cjk_fraction, tokenize  # noqa: E402


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


class TestTokenizeCjkMode(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIn("cjk", TOKENIZER_MODES)

    def test_mixed_text_bigrams_and_words(self) -> None:
        self.assertEqual(tokenize("使用 CPU 運算", "cjk"), ["使用", "cpu", "運算"])
        self.assertEqual(tokenize("使用CPU運算", "cjk"), ["使用", "cpu", "運算"])

    def test_bigrams_and_single_char_run(self) -> None:
        self.assertEqual(tokenize("疊代器", "cjk"), ["疊代", "代器"])
        self.assertEqual(tokenize("類", "cjk"), ["類"])

    def test_ascii_matches_plain(self) -> None:
        text = "Hello World_x foo-bar 42"
        self.assertEqual(tokenize(text, "cjk"), tokenize(text, "plain"))

    def test_fullwidth_latin_folded_and_punctuation_dropped(self) -> None:
        self.assertEqual(tokenize("ＣＰＵ「使用」，。", "cjk"), ["cpu", "使用"])

    def test_katakana_and_extension_a(self) -> None:
        self.assertEqual(tokenize("カタカナ", "cjk"), ["カタ", "タカ", "カナ"])
        self.assertEqual(tokenize("\u3400\u3401", "cjk"), ["\u3400\u3401"])

    def test_plain_mode_still_one_token_per_sentence(self) -> None:
        self.assertEqual(tokenize("疊代器型別", "plain"), ["疊代器型別"])

    def test_cjk_fraction(self) -> None:
        self.assertEqual(cjk_fraction(["abc"]), 0.0)
        self.assertEqual(cjk_fraction([]), 0.0)
        self.assertAlmostEqual(cjk_fraction(["使用 CPU"]), 0.4)

    def test_index_ranks_chinese_doc_and_roundtrips(self) -> None:
        docs = ["疊代器型別 Iterator Types", "logging configuration root logger"]
        index = TfidfIndex(tokenizer="cjk")
        index.fit(docs)
        self.assertEqual(index.query("疊代器")[0][0], 0)
        restored = TfidfIndex.from_dict(index.to_dict())
        self.assertEqual(restored.tokenizer, "cjk")
        self.assertEqual(restored.query("疊代器")[0][0], 0)


if __name__ == "__main__":
    unittest.main()
