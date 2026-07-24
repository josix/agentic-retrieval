"""Reciprocal Rank Fusion for combining multiple ranked lists."""

from typing import List, Optional, Tuple

#: Candidate-pool sizing constants (see ``candidate_pool``): each retriever
#: is asked for ``top_k * POOL_MULTIPLIER`` candidates (floored at
#: ``POOL_FLOOR``) before RRF fusion truncates to ``top_k``, so fusion has
#: enough per-arm overlap to work with instead of starving on a shallow pool.
POOL_MULTIPLIER = 10
POOL_FLOOR = 50


def candidate_pool(top_k: int, n_units: Optional[int] = None) -> int:
    """Return the per-retriever candidate-pool depth for a ``top_k`` query.

    ``min(max(top_k * POOL_MULTIPLIER, POOL_FLOOR), n_units)`` — deep enough
    for RRF to have real overlap to fuse, but never deeper than the corpus
    itself when *n_units* (the retriever's own indexed unit count) is known.
    ``n_units=None`` (the default) skips that cap.
    """
    pool = max(top_k * POOL_MULTIPLIER, POOL_FLOOR)
    if n_units is not None:
        pool = min(pool, n_units)
    return pool


def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists using (optionally weighted) Reciprocal Rank
    Fusion (RRF).

    Parameters
    ----------
    rankings :
        Each inner list is an ordered list of document indices (best first).
        Rank is 0-based within each list.
    k :
        Constant that dampens the impact of high ranks (default 60).
    weights :
        Optional per-ranking weight, same length/order as *rankings*; each
        ranking's contribution becomes ``weight * 1 / (k + rank + 1)``.
        ``None`` (the default) weights every ranking ``1.0``, which is
        numerically identical to the unweighted fusion.

    Returns
    -------
    List of (doc_idx, fused_score) sorted by fused_score descending.
    """
    resolved_weights = weights if weights is not None else [1.0] * len(rankings)
    scores: dict = {}
    for ranked_list, weight in zip(rankings, resolved_weights):
        for rank, doc_idx in enumerate(ranked_list):
            # rank is 0-based; RRF score = weight / (k + rank + 1) to be 1-based
            scores[doc_idx] = scores.get(doc_idx, 0.0) + weight / (k + rank + 1)

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result
