"""Tests for retrieval.eval: pure metric functions plus an end-to-end run
over the in-repo eval_corpus/eval_queries.json fixture.

The end-to-end tests never assert exact recall/nDCG values for
extras-dependent retrievers (turbovec/pi-serini/hybrid/treesitter) or any
latency magnitude/ordering — only structural invariants that hold whether
or not those extras are installed.
"""

import contextlib
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.cli import main  # noqa: E402
from retrieval.document import SearchHit  # noqa: E402
from retrieval.eval import (  # noqa: E402
    RelevantSpan,
    confidence_validity,
    format_report_text,
    ndcg_at_k,
    recall_at_k,
    report_to_dict,
    run_eval,
    span_overlaps,
)

try:
    import turbovec  # noqa: F401

    _TURBOVEC_INSTALLED = True
except ImportError:
    _TURBOVEC_INSTALLED = False

try:
    import tree_sitter_language_pack  # noqa: F401

    _TREESITTER_INSTALLED = True
except ImportError:
    _TREESITTER_INSTALLED = False

_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
_QUERIES_PATH = _FIXTURES_DIR / "eval_queries.json"
_CORPUS_DIR = _FIXTURES_DIR / "eval_corpus"


def _hit(source_path: str, start: int, end: int) -> SearchHit:
    return SearchHit(docid=f"{source_path}:{start}-{end}", source_path=source_path,
                      start_line=start, end_line=end, rank=0)


class TestSpanOverlaps(unittest.TestCase):
    def test_overlapping_spans_match(self) -> None:
        hit = _hit("a.txt", 3, 6)
        gold = RelevantSpan(source_path="a.txt", start_line=5, end_line=8)
        self.assertTrue(span_overlaps(hit, gold))

    def test_adjacent_non_overlapping_spans_do_not_match(self) -> None:
        hit = _hit("a.txt", 1, 3)
        gold = RelevantSpan(source_path="a.txt", start_line=4, end_line=6)
        self.assertFalse(span_overlaps(hit, gold))

    def test_disjoint_spans_do_not_match(self) -> None:
        hit = _hit("a.txt", 1, 2)
        gold = RelevantSpan(source_path="a.txt", start_line=10, end_line=12)
        self.assertFalse(span_overlaps(hit, gold))

    def test_different_source_path_never_matches(self) -> None:
        hit = _hit("a.txt", 1, 5)
        gold = RelevantSpan(source_path="b.txt", start_line=1, end_line=5)
        self.assertFalse(span_overlaps(hit, gold))

    def test_none_span_falls_back_to_source_path_only(self) -> None:
        hit = SearchHit(docid="a.txt", source_path="a.txt", start_line=None,
                         end_line=None, rank=0)
        gold = RelevantSpan(source_path="a.txt", start_line=5, end_line=8)
        self.assertTrue(span_overlaps(hit, gold))


class TestRecallAtK(unittest.TestCase):
    def test_recall_counts_distinct_golds_hit_in_top_k(self) -> None:
        hits = [_hit("a.txt", 1, 2), _hit("a.txt", 10, 12), _hit("a.txt", 20, 22)]
        golds = [
            RelevantSpan(source_path="a.txt", start_line=1, end_line=2),
            RelevantSpan(source_path="a.txt", start_line=10, end_line=12),
            RelevantSpan(source_path="a.txt", start_line=99, end_line=100),
        ]
        self.assertAlmostEqual(recall_at_k(hits, golds, k=3), 2 / 3)

    def test_recall_ignores_hits_beyond_k(self) -> None:
        hits = [_hit("a.txt", 1, 2), _hit("a.txt", 10, 12)]
        golds = [RelevantSpan(source_path="a.txt", start_line=10, end_line=12)]
        self.assertAlmostEqual(recall_at_k(hits, golds, k=1), 0.0)
        self.assertAlmostEqual(recall_at_k(hits, golds, k=2), 1.0)

    def test_recall_with_no_golds_is_zero(self) -> None:
        hits = [_hit("a.txt", 1, 2)]
        self.assertEqual(recall_at_k(hits, [], k=5), 0.0)

    def test_recall_dedups_multiple_hits_on_same_gold(self) -> None:
        # Two hits both overlap the single gold: still counts as one match.
        hits = [_hit("a.txt", 1, 5), _hit("a.txt", 3, 7)]
        golds = [RelevantSpan(source_path="a.txt", start_line=4, end_line=4)]
        self.assertAlmostEqual(recall_at_k(hits, golds, k=2), 1.0)


class TestNdcgAtK(unittest.TestCase):
    def test_ndcg_perfect_ranking_is_one(self) -> None:
        hits = [_hit("a.txt", 1, 2), _hit("a.txt", 10, 12)]
        golds = [
            RelevantSpan(source_path="a.txt", start_line=1, end_line=2),
            RelevantSpan(source_path="a.txt", start_line=10, end_line=12),
        ]
        self.assertAlmostEqual(ndcg_at_k(hits, golds, k=2), 1.0)

    def test_ndcg_hand_computed_value(self) -> None:
        # Only the 2nd-ranked hit is relevant: DCG = 1/log2(3); IDCG (single
        # gold) = 1/log2(2).
        hits = [_hit("a.txt", 90, 92), _hit("a.txt", 1, 2)]
        golds = [RelevantSpan(source_path="a.txt", start_line=1, end_line=2)]
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(ndcg_at_k(hits, golds, k=2), expected)

    def test_ndcg_with_no_golds_is_zero(self) -> None:
        hits = [_hit("a.txt", 1, 2)]
        self.assertEqual(ndcg_at_k(hits, [], k=5), 0.0)

    def test_ndcg_matches_each_gold_at_most_once(self) -> None:
        # Two hits overlap the same single gold; nDCG must not double-count it.
        hits = [_hit("a.txt", 1, 5), _hit("a.txt", 3, 7)]
        golds = [RelevantSpan(source_path="a.txt", start_line=4, end_line=4)]
        expected = (1.0 / math.log2(2)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(ndcg_at_k(hits, golds, k=2), expected)


class TestConfidenceValidity(unittest.TestCase):
    def test_bucket_precisions_and_monotonic_true(self) -> None:
        pairs = [
            ("high", True), ("high", True), ("high", False),
            ("medium", True), ("medium", False),
            ("low", False), ("low", False),
        ]
        result = confidence_validity(pairs)
        self.assertAlmostEqual(result["high"]["precision"], 2 / 3)
        self.assertEqual(result["high"]["relevant"], 2)
        self.assertEqual(result["high"]["total"], 3)
        self.assertAlmostEqual(result["medium"]["precision"], 0.5)
        self.assertAlmostEqual(result["low"]["precision"], 0.0)
        self.assertTrue(result["monotonic"])

    def test_non_monotonic_is_detected(self) -> None:
        pairs = [("high", False), ("medium", True), ("low", False)]
        result = confidence_validity(pairs)
        self.assertFalse(result["monotonic"])

    def test_empty_bucket_is_skipped_from_monotonic_check(self) -> None:
        pairs = [("high", True), ("low", False)]
        result = confidence_validity(pairs)
        self.assertIsNone(result["medium"]["precision"])
        self.assertEqual(result["medium"]["total"], 0)
        # high (1.0) >= low (0.0): still monotonic, medium skipped entirely.
        self.assertTrue(result["monotonic"])

    def test_empty_pairs_is_vacuously_monotonic(self) -> None:
        result = confidence_validity([])
        self.assertTrue(result["monotonic"])
        for bucket in ("high", "medium", "low"):
            self.assertIsNone(result[bucket]["precision"])


class TestQuerySetIntegrity(unittest.TestCase):
    """Every gold span in eval_queries.json must resolve to real file:line
    ranges under eval_corpus/, independent of any retriever's behavior."""

    def test_every_gold_source_path_and_span_is_within_file_bounds(self) -> None:
        payload = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["corpus_root"], "eval_corpus")
        for query in payload["queries"]:
            for span in query["relevant"]:
                file_path = _CORPUS_DIR / span["source_path"]
                with self.subTest(query=query["id"], source_path=span["source_path"]):
                    self.assertTrue(file_path.is_file(), f"{file_path} does not exist")
                    line_count = len(file_path.read_text(encoding="utf-8").splitlines())
                    self.assertGreaterEqual(span["start_line"], 1)
                    self.assertLessEqual(span["end_line"], line_count)
                    self.assertLessEqual(span["start_line"], span["end_line"])

    def test_every_query_has_a_known_category(self) -> None:
        payload = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
        categories = {q["category"] for q in payload["queries"]}
        self.assertTrue(categories <= {"vocab-mismatch", "exact-keyword", "code"})
        self.assertTrue(categories)


class TestRunEval(unittest.TestCase):
    def setUp(self) -> None:
        self.report = run_eval(_QUERIES_PATH, k=5, warm_runs=1)

    def test_report_has_one_query_eval_per_labeled_query(self) -> None:
        payload = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(self.report.queries), len(payload["queries"]))

    def test_lexical_always_present_in_aggregate(self) -> None:
        # lexical needs zero optional extras, so it must always survive.
        self.assertIn("lexical", self.report.aggregate)
        self.assertIn("consolidated", self.report.aggregate)

    def test_missing_extras_show_up_in_skipped_not_a_crash(self) -> None:
        skipped_names = {note["name"] for note in self.report.skipped}
        if not _TURBOVEC_INSTALLED:
            self.assertIn("turbovec", skipped_names)
            self.assertIn("hybrid", skipped_names)
        if not _TREESITTER_INSTALLED:
            self.assertIn("treesitter", skipped_names)

    def test_latency_fields_are_non_negative(self) -> None:
        for query_eval in self.report.queries:
            for retriever_eval in query_eval.per_retriever.values():
                self.assertGreaterEqual(retriever_eval.cold_search_s, 0.0)
                self.assertGreaterEqual(retriever_eval.warm_search_s, 0.0)

    def test_recall_and_ndcg_are_within_unit_range(self) -> None:
        for query_eval in self.report.queries:
            for retriever_eval in query_eval.per_retriever.values():
                self.assertGreaterEqual(retriever_eval.recall_at_k, 0.0)
                self.assertLessEqual(retriever_eval.recall_at_k, 1.0)
                self.assertGreaterEqual(retriever_eval.ndcg_at_k, 0.0)
                self.assertLessEqual(retriever_eval.ndcg_at_k, 1.0)

    def test_confidence_validity_buckets_present(self) -> None:
        for bucket in ("high", "medium", "low"):
            self.assertIn(bucket, self.report.confidence_validity)
        self.assertIn("monotonic", self.report.confidence_validity)

    def test_report_to_dict_round_trips_through_json(self) -> None:
        envelope = report_to_dict(self.report)
        reparsed = json.loads(json.dumps(envelope))
        self.assertEqual(reparsed["k"], 5)
        self.assertIn("lexical", reparsed["aggregate"])
        self.assertIsInstance(reparsed["skipped"], list)
        self.assertIsInstance(reparsed["build_s"], dict)

    def test_format_report_text_mentions_lexical(self) -> None:
        text = format_report_text(self.report)
        self.assertIn("lexical", text)
        self.assertIn("Confidence-signal validity", text)

    def test_root_override_is_honored(self) -> None:
        report = run_eval(_QUERIES_PATH, root=_CORPUS_DIR, k=5, warm_runs=0)
        self.assertIn("lexical", report.aggregate)

    def test_no_pi_serini_tempdir_leaks_after_run_eval(self) -> None:
        # Whether or not pyserini is installed, run_eval must leave no
        # "retrieval_eval_*" Lucene index dir behind under the OS temp root.
        before = set(pathlib.Path(tempfile.gettempdir()).glob("retrieval_eval_*"))
        run_eval(_QUERIES_PATH, k=5, warm_runs=0)
        after = set(pathlib.Path(tempfile.gettempdir()).glob("retrieval_eval_*"))
        self.assertEqual(before, after)


class TestEvalCli(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_eval_subcommand_json_envelope_shape(self) -> None:
        code, out = self._run(
            ["eval", "--queries", str(_QUERIES_PATH), "--json", "--warm-runs", "0"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(
            set(payload.keys()),
            {"k", "queries", "aggregate", "confidence_validity", "skipped", "build_s"},
        )
        self.assertIn("lexical", payload["aggregate"])

    def test_eval_subcommand_text_mode(self) -> None:
        code, out = self._run(
            ["eval", "--queries", str(_QUERIES_PATH), "--warm-runs", "0"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Eval report", out)

    def test_eval_subcommand_writes_output_file(self) -> None:
        with contextlib.ExitStack() as stack:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            output_path = pathlib.Path(tmp_dir) / "report.json"
            code, _out = self._run(
                [
                    "eval", "--queries", str(_QUERIES_PATH),
                    "--warm-runs", "0", "--output", str(output_path),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("aggregate", payload)

    def test_eval_subcommand_missing_queries_arg_exits_nonzero(self) -> None:
        with self.assertRaises(SystemExit):
            main(["eval"])


@unittest.skipUnless(_TURBOVEC_INSTALLED, "turbovec not installed")
class TestRunEvalWithTurbovec(unittest.TestCase):
    def test_turbovec_and_hybrid_are_not_skipped(self) -> None:
        report = run_eval(_QUERIES_PATH, k=5, warm_runs=0)
        skipped_names = {note["name"] for note in report.skipped}
        self.assertNotIn("turbovec", skipped_names)
        self.assertNotIn("hybrid", skipped_names)
        self.assertIn("turbovec", report.aggregate)
        self.assertIn("hybrid", report.aggregate)


@unittest.skipUnless(_TREESITTER_INSTALLED, "tree-sitter-language-pack not installed")
class TestRunEvalWithTreesitter(unittest.TestCase):
    def test_treesitter_is_not_skipped(self) -> None:
        report = run_eval(_QUERIES_PATH, k=5, warm_runs=0)
        skipped_names = {note["name"] for note in report.skipped}
        self.assertNotIn("treesitter", skipped_names)
        self.assertIn("treesitter", report.aggregate)


if __name__ == "__main__":
    unittest.main()
