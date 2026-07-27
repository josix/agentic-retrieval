"""Tests for reciprocal_rank_fusion, including the weighted-RRF extension."""

import unittest

from retrieval.fusion import candidate_pool, reciprocal_rank_fusion


class TestReciprocalRankFusion(unittest.TestCase):
    def test_equal_weight_regression_matches_unweighted(self) -> None:
        rankings = [[0, 1, 2], [2, 0, 1]]
        unweighted = reciprocal_rank_fusion(rankings)
        weighted_default = reciprocal_rank_fusion(rankings, weights=None)
        weighted_all_ones = reciprocal_rank_fusion(rankings, weights=[1.0, 1.0])
        self.assertEqual(unweighted, weighted_default)
        self.assertEqual(unweighted, weighted_all_ones)

        # Pin down the exact numbers so a future refactor can't silently
        # change the fusion math for existing (unweighted) callers.
        scores = dict(unweighted)
        k = 60
        self.assertAlmostEqual(scores[0], 1 / (k + 1) + 1 / (k + 2))
        self.assertAlmostEqual(scores[1], 1 / (k + 2) + 1 / (k + 3))
        self.assertAlmostEqual(scores[2], 1 / (k + 3) + 1 / (k + 1))

    def test_weighted_bias_favors_the_higher_weighted_list(self) -> None:
        # List A ranks doc 0 first; list B ranks doc 1 first. Weighting B
        # heavily must flip the winner from 0 to 1.
        rankings = [[0, 1], [1, 0]]
        fused = reciprocal_rank_fusion(rankings, weights=[0.1, 5.0])
        self.assertEqual(fused[0][0], 1)

    def test_deterministic_tie_order(self) -> None:
        # Two candidates with identical fused scores: sorted() is stable, so
        # ties keep the original (dict-insertion) order across repeated runs.
        rankings = [[0, 1]]
        first = reciprocal_rank_fusion(rankings)
        second = reciprocal_rank_fusion(rankings)
        self.assertEqual(first, second)

    def test_empty_rankings_returns_empty_list(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_single_list_preserves_relative_order(self) -> None:
        fused = reciprocal_rank_fusion([[5, 2, 9]])
        self.assertEqual([idx for idx, _score in fused], [5, 2, 9])


class TestCandidatePool(unittest.TestCase):
    def test_floor_applies_for_small_top_k(self) -> None:
        self.assertEqual(candidate_pool(5), 50)

    def test_capped_by_n_units(self) -> None:
        self.assertEqual(candidate_pool(5, 20), 20)

    def test_multiplier_applies_above_the_floor(self) -> None:
        self.assertEqual(candidate_pool(10), 100)

    def test_n_units_above_pool_does_not_cap(self) -> None:
        self.assertEqual(candidate_pool(10, 1000), 100)


if __name__ == "__main__":
    unittest.main()
