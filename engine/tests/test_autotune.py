"""Tests for retrieval.autotune's signal collection and hyperparameter
resolution heuristics.

Threshold-crossing behavior (code_fraction, tokenizer mode, k1's top
branches) is exercised against real tmpdir fixtures via ``collect_signals``;
the "b" length-normalization branches and the anchor test are exercised
against hand-built ``CorpusSignals`` instances instead of real files, since
hitting an exact chunk-token coefficient-of-variation through real chunking
is fragile and unnecessary to prove the formula itself.
"""

import pathlib
import sys
import tempfile
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.autotune import CorpusSignals, collect_signals, resolve_params  # noqa: E402


def _write(path: pathlib.Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestCollectSignalsProse(unittest.TestCase):
    def test_all_prose_corpus_has_zero_code_fraction_and_plain_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "a.md", "# Hello\n\nSome prose about routing and networks.\n" * 5)
            _write(root / "b.txt", "More prose, no code at all.\n" * 5)
            signals = collect_signals(root)
            self.assertEqual(signals.code_fraction, 0.0)
            params = resolve_params(signals)
            self.assertEqual(params["tokenizer"], "plain")
            self.assertEqual(params["code_chars"], 400)


class TestCollectSignalsCode(unittest.TestCase):
    def test_all_python_corpus_is_code_tokenizer_and_low_k1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            body = "\n\n".join(f"def fn_{i}():\n    return {i} * 2\n" for i in range(40))
            _write(root / "app.py", body)
            _write(root / "lib.py", body)
            signals = collect_signals(root)
            self.assertGreater(signals.code_fraction, 0.6)
            params = resolve_params(signals)
            self.assertEqual(params["tokenizer"], "code")
            self.assertEqual(params["bm25_k1"], 1.2)
            self.assertEqual(params["code_chars"], 1200)


class TestResolveParamsLucene(unittest.TestCase):
    """lucene_k1/lucene_b are always explicitly derived (never omitted),
    even though today's chunkers never produce a chunk long enough to hit
    the long-document branch."""

    def test_short_median_chunk_yields_ms_marco_tuning(self) -> None:
        signals = CorpusSignals(n_files=5, code_fraction=0.1, p50_chunk_tokens=100.0)
        params = resolve_params(signals)
        self.assertEqual(params["lucene_k1"], 0.9)
        self.assertEqual(params["lucene_b"], 0.4)

    def test_unknown_median_chunk_falls_back_to_ms_marco_tuning(self) -> None:
        params = resolve_params(CorpusSignals(n_files=0))
        self.assertEqual(params["lucene_k1"], 0.9)
        self.assertEqual(params["lucene_b"], 0.4)

    def test_long_median_chunk_yields_long_document_tuning(self) -> None:
        signals = CorpusSignals(n_files=5, code_fraction=0.1, p50_chunk_tokens=2500.0)
        params = resolve_params(signals)
        self.assertEqual(params["lucene_k1"], 25.0)
        self.assertEqual(params["lucene_b"], 1.0)


class TestResolveParamsBFormula(unittest.TestCase):
    """The bm25_b heuristic reads chunk_token_cv/p50/p90 — exercised
    directly on synthetic CorpusSignals rather than through real chunking."""

    def _signals(self, cv: float, p50: float, p90: float) -> CorpusSignals:
        return CorpusSignals(
            n_files=5,
            code_fraction=0.1,
            n_chunks=20,
            mean_chunk_tokens=100.0,
            chunk_token_cv=cv,
            p50_chunk_tokens=p50,
            p90_chunk_tokens=p90,
        )

    def test_low_cv_yields_low_b(self) -> None:
        params = resolve_params(self._signals(cv=0.2, p50=100, p90=150))
        self.assertEqual(params["bm25_b"], 0.3)

    def test_mid_cv_yields_mid_b(self) -> None:
        params = resolve_params(self._signals(cv=0.6, p50=100, p90=200))
        self.assertEqual(params["bm25_b"], 0.75)

    def test_high_cv_yields_high_b(self) -> None:
        params = resolve_params(self._signals(cv=1.5, p50=100, p90=200))
        self.assertEqual(params["bm25_b"], 1.0)

    def test_high_spread_ratio_yields_high_b_even_at_mid_cv(self) -> None:
        params = resolve_params(self._signals(cv=0.6, p50=50, p90=300))  # ratio 6 > 4
        self.assertEqual(params["bm25_b"], 1.0)

    def test_empty_corpus_falls_back_to_static_default_b(self) -> None:
        params = resolve_params(CorpusSignals(n_files=0))
        self.assertEqual(params["bm25_b"], 0.75)
        self.assertEqual(params["bit_width"], 4)
        self.assertEqual(params["bm25_k1"], 1.5)


class TestResolveParamsEmbedModel(unittest.TestCase):
    """model_name is decided from corpus content (code fraction) and size
    (chunk count), coupled to bit_width via the chosen model's dims."""

    def test_code_heavy_corpus_picks_code_search_model(self) -> None:
        signals = CorpusSignals(n_files=5, code_fraction=0.8, n_chunks=100)
        params = resolve_params(signals)
        self.assertEqual(
            params["model_name"],
            "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        )

    def test_small_prose_corpus_affords_quality_model(self) -> None:
        signals = CorpusSignals(n_files=5, code_fraction=0.1, n_chunks=100)
        params = resolve_params(signals)
        self.assertEqual(params["model_name"], "sentence-transformers/all-mpnet-base-v2")

    def test_large_prose_corpus_keeps_fast_default(self) -> None:
        signals = CorpusSignals(n_files=500, code_fraction=0.1, n_chunks=50_000)
        params = resolve_params(signals)
        self.assertEqual(params["model_name"], "sentence-transformers/all-MiniLM-L6-v2")

    def test_empty_corpus_falls_back_to_static_default_model(self) -> None:
        params = resolve_params(CorpusSignals(n_files=0))
        self.assertEqual(params["model_name"], "sentence-transformers/all-MiniLM-L6-v2")

    def test_bit_width_sized_from_chosen_model_dims(self) -> None:
        # 100_000 chunks: * 384 dims = 3.84e7 <= 5e7 -> 4, but the small-
        # corpus threshold is exceeded so MiniLM (384) applies; with a
        # 768-dim override the same corpus crosses into the 3-bit tier.
        signals = CorpusSignals(n_files=500, code_fraction=0.1, n_chunks=100_000)
        self.assertEqual(resolve_params(signals)["bit_width"], 4)
        overridden = resolve_params(
            signals, overrides={"model_name": "sentence-transformers/all-mpnet-base-v2"}
        )
        self.assertEqual(overridden["model_name"], "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(overridden["bit_width"], 3)


class TestResolveParamsOverrides(unittest.TestCase):
    def test_explicit_overrides_always_win(self) -> None:
        signals = CorpusSignals(n_files=3, code_fraction=0.9, n_chunks=10, mean_chunk_tokens=200.0)
        params = resolve_params(signals, overrides={"bm25_k1": 42.0, "tokenizer": "plain"})
        self.assertEqual(params["bm25_k1"], 42.0)
        self.assertEqual(params["tokenizer"], "plain")

    def test_none_valued_overrides_do_not_win(self) -> None:
        signals = CorpusSignals(n_files=1, code_fraction=0.0)
        params = resolve_params(signals, overrides={"bm25_k1": None})
        self.assertEqual(params["bm25_k1"], 1.5)


class TestDeterminism(unittest.TestCase):
    def test_collect_signals_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "a.py", "def f():\n    return 1\n")
            _write(root / "b.md", "# doc\n\nsome text\n")
            first = collect_signals(root)
            second = collect_signals(root)
            self.assertEqual(first, second)

    def test_resolve_params_is_deterministic(self) -> None:
        signals = CorpusSignals(n_files=2, code_fraction=0.5, n_chunks=5, mean_chunk_tokens=80.0)
        self.assertEqual(resolve_params(signals), resolve_params(signals))


class TestAutotunePdfByteAccounting(unittest.TestCase):
    """T-A1: a large PDF dropped into a code-dominated corpus must not flip
    the corpus's code-vs-prose signals — ``collect_signals`` substitutes the
    extracted sidecar transcript's (small) length for the PDF's raw byte
    size when accounting ``total_bytes``, so a bulky PDF never dilutes
    ``code_fraction`` the way its raw size would."""

    def test_large_pdf_does_not_flip_code_dominated_signals(self) -> None:
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            body = "\n\n".join(f"def fn_{i}():\n    return {i} * 2\n" for i in range(60))
            _write(root / "app.py", body)
            _write(root / "lib.py", body)
            baseline = collect_signals(root)
            self.assertGreater(baseline.code_fraction, 0.6)

            _write_pdf(root / "manual.pdf", pages=30)
            with_pdf = collect_signals(root)
            self.assertGreater(with_pdf.code_fraction, 0.6)
            self.assertEqual(resolve_params(with_pdf)["tokenizer"], "code")
            self.assertEqual(resolve_params(with_pdf)["code_chars"], 1200)


class TestAnchorSyntheticCorpus(unittest.TestCase):
    """A synthetic (hand-built) CorpusSignals landing in the "typical mixed
    repo" band this engine's own tuning aims at: mostly-code with some docs,
    moderate chunk-size variance, small corpus."""

    def test_mixed_repo_signals_resolve_to_expected_band(self) -> None:
        signals = CorpusSignals(
            n_files=50,
            code_fraction=0.4,
            n_chunks=500,
            mean_chunk_tokens=120.0,
            chunk_token_cv=0.6,
            p50_chunk_tokens=110.0,
            p90_chunk_tokens=250.0,
            median_code_lines=200.0,
        )
        params = resolve_params(signals)
        self.assertEqual(params["tokenizer"], "code")
        self.assertEqual(params["bm25_k1"], 1.5)
        self.assertEqual(params["bm25_b"], 0.75)
        self.assertEqual(params["code_chars"], 1200)
        self.assertEqual(params["bit_width"], 4)


if __name__ == "__main__":
    unittest.main()
