"""Reciprocal Rank Fusion for combining multiple ranked lists."""

from typing import List, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    rankings :
        Each inner list is an ordered list of document indices (best first).
        Rank is 0-based within each list.
    k :
        Constant that dampens the impact of high ranks (default 60).

    Returns
    -------
    List of (doc_idx, fused_score) sorted by fused_score descending.
    """
    scores: dict = {}
    for ranked_list in rankings:
        for rank, doc_idx in enumerate(ranked_list):
            # rank is 0-based; RRF score = 1 / (k + rank + 1) to be 1-based
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result
