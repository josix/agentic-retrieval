"""Consolidate per-retriever ``SearchHit`` rankings into a single, deduplicated,
explainable ranking a following conversation/agent can act on directly.

Pure stdlib. Given ``{retriever_name: [SearchHit, ...]}`` (one ranked list per
retriever already run against the same query), ``consolidate`` merges hits
whose spans overlap (or are line-adjacent) into a single canonical
``ConsolidatedHit``, fuses each group's per-retriever ranks with a (optionally
weighted) Reciprocal Rank Fusion, and returns the fused list sorted best
first — with provenance (which retrievers found it), agreement (how many),
and a confidence label baked in so the handoff is self-explanatory without
re-deriving any of this from the raw per-retriever output.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from retrieval.document import SearchHit
from retrieval.fusion import reciprocal_rank_fusion

#: Fixed, reproducible retriever ordering: known strategies first (in their
#: canonical CLI order), any extra/unknown retriever name appended
#: alphabetically. Consumed by ``consolidate`` so weight mapping and
#: iteration order never depend on the caller's dict ordering.
_CANONICAL_ORDER = ("lexical", "turbovec", "pi-serini", "hybrid", "treesitter")

#: Retriever names whose sole-contributor confidence is "medium" rather than
#: "low" (dense/Lucene arms are considered strong single-arm signals).
_STRONG_SOLO_RETRIEVERS = frozenset({"turbovec", "pi-serini"})


@dataclass
class ConsolidatedHit:
    """One deduplicated, ranked, explainable result in the consolidated list."""

    source_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    docid: str
    context: str
    score: float
    rank: int
    provenance: List[str] = field(default_factory=list)
    agreement: int = 0
    confidence: str = "low"
    contributors: List[dict] = field(default_factory=list)


def _ordered_retriever_names(names) -> List[str]:
    """Canonical order: known strategies first, extras alphabetically after."""
    known = [n for n in _CANONICAL_ORDER if n in names]
    extra = sorted(n for n in names if n not in _CANONICAL_ORDER)
    return known + extra


def _spans_overlap_or_adjacent(a: SearchHit, b: SearchHit, merge_adjacent: bool) -> bool:
    """Whether two same-path hits' ``[start, end]`` spans overlap, or are
    line-adjacent (``a.end + 1 == b.start``) when *merge_adjacent*."""
    a_start, a_end = a.start_line, a.end_line
    b_start, b_end = b.start_line, b.end_line
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    if a_start <= b_end and b_start <= a_end:
        return True
    if merge_adjacent:
        return a_end + 1 == b_start or b_end + 1 == a_start
    return False


def _group_key(hit: SearchHit) -> str:
    """Grouping key for hits without a line span (kept as non-overlapping
    singletons, one group per distinct docid)."""
    return hit.docid


def _group_hits_by_span(hits: List[SearchHit], merge_adjacent: bool) -> List[List[SearchHit]]:
    """Group same-``source_path`` hits whose spans overlap/are adjacent into a
    single group; hits with no line span form their own singleton group keyed
    by docid (never merged with anything else)."""
    by_path: Dict[str, List[SearchHit]] = {}
    singletons: List[List[SearchHit]] = []
    seen_singleton_keys: Dict[str, int] = {}
    for hit in hits:
        if hit.start_line is None or hit.end_line is None:
            key = _group_key(hit)
            if key in seen_singleton_keys:
                singletons[seen_singleton_keys[key]].append(hit)
            else:
                seen_singleton_keys[key] = len(singletons)
                singletons.append([hit])
            continue
        by_path.setdefault(hit.source_path, []).append(hit)

    groups: List[List[SearchHit]] = list(singletons)
    for path_hits in by_path.values():
        groups.extend(_merge_overlapping_spans(path_hits, merge_adjacent))
    return groups


def _merge_overlapping_spans(
    path_hits: List[SearchHit], merge_adjacent: bool
) -> List[List[SearchHit]]:
    """Union-merge overlapping/adjacent spans within a single source path."""
    ordered = sorted(path_hits, key=lambda h: (h.start_line, h.end_line, h.docid))
    groups: List[List[SearchHit]] = []
    for hit in ordered:
        merged = False
        for group in groups:
            if any(_spans_overlap_or_adjacent(hit, member, merge_adjacent) for member in group):
                group.append(hit)
                merged = True
                break
        if not merged:
            groups.append([hit])
    return groups


def _canonical_member(group: List[SearchHit], rank_lookup: Dict[int, Dict[str, int]],
                       group_index: int) -> SearchHit:
    """Pick the group's canonical member: the contributor with the best
    (lowest) single-arm rank, deterministically tie-broken by
    ``(start_line, end_line, docid)``."""
    def sort_key(hit: SearchHit):
        arm_rank = rank_lookup[group_index].get(id(hit), hit.rank)
        return (arm_rank, hit.start_line or 0, hit.end_line or 0, hit.docid)

    return min(group, key=sort_key)


def _canonical_context(group: List[SearchHit], ordered_names: List[str],
                        retriever_of: Dict[int, str]) -> str:
    """Canonical ``context``: falls back to the first non-empty member's
    context, iterated in fixed canonical retriever order for reproducibility."""
    by_name = {retriever_of[id(hit)]: hit for hit in group}
    for name in ordered_names:
        hit = by_name.get(name)
        if hit is not None and hit.context:
            return hit.context
    return ""


def _confidence_for(agreement: int, provenance: List[str]) -> str:
    """"high" if >=2 retrievers agree; else "medium" if the sole contributing
    arm is a dense/Lucene arm; else "low"."""
    if agreement >= 2:
        return "high"
    if provenance and provenance[0] in _STRONG_SOLO_RETRIEVERS:
        return "medium"
    return "low"


def consolidate(
    per_retriever_hits: Dict[str, List[SearchHit]],
    *,
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    merge_adjacent: bool = True,
) -> List[ConsolidatedHit]:
    """Merge, fuse, and rank per-retriever ``SearchHit`` lists into a single
    deduplicated, explainable ``ConsolidatedHit`` list.

    Parameters
    ----------
    per_retriever_hits :
        ``{retriever_name: [SearchHit, ...]}``, one ranked list per retriever
        already run against the same query. Retrievers may be a subset of
        the six strategies (graceful skip) and may be supplied in any dict
        order — iteration is always in fixed canonical order internally, so
        the result is reproducible regardless of input dict order.
    k :
        RRF damping constant, forwarded to ``reciprocal_rank_fusion``.
    weights :
        Optional ``{retriever_name: weight}``; retrievers absent from this
        mapping default to weight ``1.0``.
    merge_adjacent :
        When ``True`` (default), same-path spans whose end/start lines are
        exactly adjacent (``end + 1 == start``) are merged into one group in
        addition to spans that overlap.

    Returns
    -------
    ``ConsolidatedHit`` list sorted by fused score descending (ties broken by
    ``source_path``, ``start_line``, ``end_line``, ``docid``), with ``rank``
    assigned 1-based over that final order.
    """
    ordered_names = _ordered_retriever_names(per_retriever_hits.keys())
    all_hits: List[SearchHit] = []
    retriever_of: Dict[int, str] = {}
    for name in ordered_names:
        for hit in per_retriever_hits[name]:
            all_hits.append(hit)
            retriever_of[id(hit)] = name

    groups = _group_hits_by_span(all_hits, merge_adjacent)

    # Per-group, per-retriever best (lowest) single-arm rank -> group index.
    group_index_of_hit: Dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for hit in group:
            group_index_of_hit[id(hit)] = group_index

    rankings, resolved_weights = _build_rankings(
        ordered_names, per_retriever_hits, group_index_of_hit, weights
    )
    fused = reciprocal_rank_fusion(rankings, k=k, weights=resolved_weights)

    rank_lookup = _rank_lookup_by_group(ordered_names, per_retriever_hits, group_index_of_hit)
    return _assemble_consolidated_hits(fused, groups, rank_lookup, ordered_names, retriever_of)


def _build_rankings(ordered_names, per_retriever_hits, group_index_of_hit, weights):
    """Build the per-retriever ``[group_idx, ...]`` rankings (best first,
    de-duplicated per retriever) plus the matching weight list."""
    rankings: List[List[int]] = []
    resolved_weights: List[float] = []
    for name in ordered_names:
        seen: set = set()
        ranking: List[int] = []
        for hit in per_retriever_hits[name]:
            group_idx = group_index_of_hit[id(hit)]
            if group_idx not in seen:
                seen.add(group_idx)
                ranking.append(group_idx)
        rankings.append(ranking)
        resolved_weights.append((weights or {}).get(name, 1.0))
    return rankings, resolved_weights


def _rank_lookup_by_group(ordered_names, per_retriever_hits, group_index_of_hit):
    """``{group_idx: {id(hit): single-arm rank}}`` used to pick each group's
    canonical member (best single-arm rank across its contributors)."""
    lookup: Dict[int, Dict[int, int]] = {}
    for name in ordered_names:
        for hit in per_retriever_hits[name]:
            group_idx = group_index_of_hit[id(hit)]
            lookup.setdefault(group_idx, {})[id(hit)] = hit.rank
    return lookup


def _assemble_consolidated_hits(fused, groups, rank_lookup, ordered_names, retriever_of):
    hits: List[ConsolidatedHit] = []
    for group_idx, score in fused:
        group = groups[group_idx]
        canonical = _canonical_member(group, rank_lookup, group_idx)
        provenance = sorted({retriever_of[id(hit)] for hit in group})
        agreement = len(provenance)
        contributors = [
            {"retriever": retriever_of[id(hit)], "rank": hit.rank, "docid": hit.docid}
            for hit in sorted(group, key=lambda h: (retriever_of[id(h)], h.rank))
        ]
        hits.append(
            ConsolidatedHit(
                source_path=canonical.source_path,
                start_line=canonical.start_line,
                end_line=canonical.end_line,
                docid=canonical.docid,
                context=_canonical_context(group, ordered_names, retriever_of),
                score=score,
                rank=0,  # assigned below, after the final sort
                provenance=provenance,
                agreement=agreement,
                confidence=_confidence_for(agreement, provenance),
                contributors=contributors,
            )
        )

    hits.sort(key=lambda h: (-h.score, h.source_path, h.start_line or 0, h.end_line or 0, h.docid))
    for rank, hit in enumerate(hits, start=1):
        hit.rank = rank
    return hits
