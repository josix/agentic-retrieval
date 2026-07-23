"""Tests for retrieval.consolidation.consolidate."""

import unittest

from retrieval.consolidation import consolidate
from retrieval.document import SearchHit


def _hit(docid, path, start, end, rank, context="") -> SearchHit:
    return SearchHit(
        docid=docid, source_path=path, start_line=start, end_line=end,
        rank=rank, context=context,
    )


class TestConsolidate(unittest.TestCase):
    def test_span_overlap_merges_same_region_hits(self) -> None:
        # lexical's line-chunk span and treesitter's AST-chunk span over the
        # same region of the same file must merge into one group.
        per_retriever = {
            "lexical": [_hit("a.py:1-10", "a.py", 1, 10, rank=0)],
            "treesitter": [_hit("a.py:2-8", "a.py", 2, 8, rank=0, context="Foo.bar")],
        }
        result = consolidate(per_retriever)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].agreement, 2)
        self.assertEqual(sorted(result[0].provenance), ["lexical", "treesitter"])

    def test_disjoint_spans_stay_separate(self) -> None:
        per_retriever = {
            "lexical": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
            "turbovec": [_hit("a.py:50-60", "a.py", 50, 60, rank=0)],
        }
        result = consolidate(per_retriever)
        self.assertEqual(len(result), 2)
        self.assertEqual({h.agreement for h in result}, {1})

    def test_provenance_agreement_confidence_high(self) -> None:
        per_retriever = {
            "lexical": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
            "turbovec": [_hit("a.py:1-5", "a.py", 1, 5, rank=1)],
            "hybrid": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
        }
        result = consolidate(per_retriever)
        self.assertEqual(result[0].agreement, 3)
        self.assertEqual(result[0].confidence, "high")

    def test_single_retriever_input_medium_confidence_for_dense_arm(self) -> None:
        per_retriever = {"turbovec": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)]}
        result = consolidate(per_retriever)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].agreement, 1)
        self.assertEqual(result[0].confidence, "medium")

    def test_single_retriever_input_low_confidence_for_lexical_arm(self) -> None:
        per_retriever = {"lexical": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)]}
        result = consolidate(per_retriever)
        self.assertEqual(result[0].confidence, "low")

    def test_zero_hits_returns_empty_list(self) -> None:
        self.assertEqual(consolidate({}), [])
        self.assertEqual(consolidate({"lexical": []}), [])

    def test_weights_effect_can_flip_ranking(self) -> None:
        per_retriever = {
            "lexical": [
                _hit("a.py:1-5", "a.py", 1, 5, rank=0),
                _hit("b.py:1-5", "b.py", 1, 5, rank=1),
            ],
            "turbovec": [
                _hit("b.py:1-5", "b.py", 1, 5, rank=0),
                _hit("a.py:1-5", "a.py", 1, 5, rank=1),
            ],
        }
        unweighted = consolidate(per_retriever)
        # Tied fused scores; unweighted result breaks ties by (path, ...) so
        # a.py comes first regardless. Heavily weighting turbovec must flip
        # the top result to b.py.
        weighted = consolidate(per_retriever, weights={"turbovec": 10.0})
        self.assertEqual(unweighted[0].source_path, "a.py")
        self.assertEqual(weighted[0].source_path, "b.py")

    def test_reproducible_under_shuffled_input_dict_order(self) -> None:
        per_retriever_a = {
            "lexical": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
            "turbovec": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
            "hybrid": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
        }
        per_retriever_b = {
            "hybrid": per_retriever_a["hybrid"],
            "lexical": per_retriever_a["lexical"],
            "turbovec": per_retriever_a["turbovec"],
        }
        result_a = consolidate(per_retriever_a)
        result_b = consolidate(per_retriever_b)
        self.assertEqual(
            [(h.docid, h.score, h.provenance) for h in result_a],
            [(h.docid, h.score, h.provenance) for h in result_b],
        )

    def test_merge_adjacent_true_merges_adjacent_spans(self) -> None:
        per_retriever = {
            "lexical": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
            "turbovec": [_hit("a.py:6-10", "a.py", 6, 10, rank=0)],
        }
        merged = consolidate(per_retriever, merge_adjacent=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].agreement, 2)

    def test_merge_adjacent_false_keeps_adjacent_spans_separate(self) -> None:
        per_retriever = {
            "lexical": [_hit("a.py:1-5", "a.py", 1, 5, rank=0)],
            "turbovec": [_hit("a.py:6-10", "a.py", 6, 10, rank=0)],
        }
        unmerged = consolidate(per_retriever, merge_adjacent=False)
        self.assertEqual(len(unmerged), 2)

    def test_none_line_hits_are_non_overlapping_singletons_keyed_by_docid(self) -> None:
        per_retriever = {
            "lexical": [_hit("d1", "d1", None, None, rank=0)],
            "turbovec": [_hit("d1", "d1", None, None, rank=0)],
            "hybrid": [_hit("d2", "d2", None, None, rank=0)],
        }
        result = consolidate(per_retriever)
        self.assertEqual(len(result), 2)
        by_docid = {h.docid: h for h in result}
        self.assertEqual(by_docid["d1"].agreement, 2)
        self.assertEqual(by_docid["d2"].agreement, 1)

    def test_canonical_member_uses_best_single_arm_rank(self) -> None:
        # treesitter ranks this chunk #0 (best); lexical ranks its
        # overlapping span #3. The canonical context/docid should come from
        # the treesitter (better-ranked) contributor.
        per_retriever = {
            "lexical": [_hit("a.py:1-10", "a.py", 1, 10, rank=3)],
            "treesitter": [_hit("a.py:2-8", "a.py", 2, 8, rank=0, context="Foo.bar")],
        }
        result = consolidate(per_retriever)
        self.assertEqual(result[0].docid, "a.py:2-8")
        self.assertEqual(result[0].context, "Foo.bar")

    def test_rank_assigned_starting_at_one(self) -> None:
        per_retriever = {
            "lexical": [
                _hit("a.py:1-5", "a.py", 1, 5, rank=0),
                _hit("b.py:1-5", "b.py", 100, 105, rank=1),
            ],
        }
        result = consolidate(per_retriever)
        self.assertEqual([h.rank for h in result], [1, 2])


if __name__ == "__main__":
    unittest.main()
