"""Evaluation harness: recall@k / nDCG@k per retriever vs the consolidated
fusion, confidence-signal validity, and cold/warm query latency.

Stdlib + ``retrieval.*`` imports only — this module must import cleanly with
zero optional extras installed. Heavy backends (``turbovec``, ``pi-serini``,
``treesitter``) are attempted per-strategy inside ``run_eval`` and skipped on
a ``RuntimeError``, exactly like ``retrieval.cli._index_all`` — never a hard
import-time dependency.

Everything here runs fully in-memory against a labeled query set (see
``docs/how-to/evaluate-retrievers.md`` for the JSON schema); no
``save_index``/on-disk cache writes happen, so running an eval never touches
a project's real ``<project-root>/.agentic-retrieval`` index.
"""

import json
import math
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

from retrieval.chunker import ChunkingPolicy
from retrieval.consolidation import consolidate
from retrieval.document import SearchHit
from retrieval.fusion import candidate_pool
from retrieval.project_loader import load_ast_chunk_documents, load_chunk_documents
from retrieval.retrievers import PiSeriniRetriever, Retriever, build_retriever, resolve_ctor_kwargs

#: Strategies attempted by run_eval, in report order (mirrors
#: ``retrieval.cli._DEFAULT_INDEX_SET``).
_STRATEGIES = ("lexical", "turbovec", "pi-serini", "hybrid", "treesitter")

_CONFIDENCE_BUCKETS = ("high", "medium", "low")

_PathLike = Union[Path, str]


@dataclass
class RelevantSpan:
    """One hand-labeled gold span: ``source_path`` relative to the corpus
    root plus its 1-based ``[start_line, end_line]``."""

    source_path: str
    start_line: Optional[int]
    end_line: Optional[int]


@dataclass
class LabeledQuery:
    """One labeled query: free-text ``query``, a ``category`` tag
    (``vocab-mismatch``/``exact-keyword``/``code``), and its gold spans."""

    id: str
    query: str
    category: str
    relevant: List[RelevantSpan]


@dataclass
class RetrieverEval:
    """One retriever's (or the consolidated fusion's) metrics for one query."""

    name: str
    recall_at_k: float
    ndcg_at_k: float
    cold_search_s: float
    warm_search_s: float


@dataclass
class QueryEval:
    """Per-query results: one ``RetrieverEval`` per strategy that ran (plus
    ``"consolidated"``), and the consolidated top-k confidence/relevance
    pairs used by ``confidence_validity``."""

    query_id: str
    category: str
    per_retriever: Dict[str, RetrieverEval] = field(default_factory=dict)
    confidence_hits: List[Tuple[str, bool]] = field(default_factory=list)


@dataclass
class EvalReport:
    """Full harness output: per-query results, per-retriever aggregate
    means, confidence-bucket precision, skipped strategies, and index
    build times."""

    k: int
    queries: List[QueryEval]
    aggregate: Dict[str, Dict[str, float]]
    confidence_validity: Dict[str, Any]
    skipped: List[Dict[str, str]]
    build_s: Dict[str, float]


def _normalize_source_path(path: str) -> str:
    """POSIX-normalize a source path for comparison; hits may carry a path
    relative to the corpus root (the common case) or an absolute one."""
    return PurePosixPath(str(path)).as_posix()


def _paths_match(hit_path: str, gold_path: str) -> bool:
    """Robust path comparison: exact match, or one is a path-suffix of the
    other (covers a hit's path being absolute while the gold label is
    corpus-root-relative, or vice versa)."""
    hit_norm = _normalize_source_path(hit_path)
    gold_norm = _normalize_source_path(gold_path)
    return (
        hit_norm == gold_norm
        or hit_norm.endswith("/" + gold_norm)
        or gold_norm.endswith("/" + hit_norm)
    )


def span_overlaps(hit: Any, gold: RelevantSpan) -> bool:
    """Whether *hit* (a ``SearchHit`` or ``ConsolidatedHit``) matches *gold*.

    Same ``source_path`` is required. If either side lacks a line span, the
    match falls back to the source path alone; otherwise the spans must
    overlap (``hit.start <= gold.end and gold.start <= hit.end``).
    """
    if not _paths_match(hit.source_path, gold.source_path):
        return False
    if hit.start_line is None or hit.end_line is None:
        return True
    if gold.start_line is None or gold.end_line is None:
        return True
    return hit.start_line <= gold.end_line and gold.start_line <= hit.end_line


def recall_at_k(hits: List[Any], golds: List[RelevantSpan], k: int) -> float:
    """Fraction of distinct *golds* overlapped by at least one of the top-k
    *hits*; ``0.0`` when *golds* is empty."""
    if not golds:
        return 0.0
    top = hits[:k]
    matched: set = set()
    for gold_index, gold in enumerate(golds):
        if any(span_overlaps(hit, gold) for hit in top):
            matched.add(gold_index)
    return len(matched) / len(golds)


def ndcg_at_k(hits: List[Any], golds: List[RelevantSpan], k: int) -> float:
    """Binary-gain nDCG@k: a hit at 1-based rank ``p`` earns gain 1 the first
    time it overlaps a not-yet-matched gold, else 0; ``0.0`` when the ideal
    DCG (``IDCG``) is zero (no golds)."""
    if not golds:
        return 0.0
    top = hits[:k]
    matched_golds: set = set()
    dcg = 0.0
    for position, hit in enumerate(top, start=1):
        for gold_index, gold in enumerate(golds):
            if gold_index in matched_golds:
                continue
            if span_overlaps(hit, gold):
                matched_golds.add(gold_index)
                dcg += 1.0 / math.log2(position + 1)
                break
    idcg = sum(1.0 / math.log2(p + 1) for p in range(1, min(k, len(golds)) + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def confidence_validity(pairs: List[Tuple[str, bool]]) -> Dict[str, Any]:
    """Per-confidence-bucket precision (``relevant / total``) over
    ``(confidence, is_relevant)`` pairs, plus a ``monotonic`` flag: whether
    precision decreases (or stays flat) from ``high`` to ``medium`` to
    ``low``, skipping any bucket with zero samples.
    """
    buckets: Dict[str, Dict[str, int]] = {
        bucket: {"relevant": 0, "total": 0} for bucket in _CONFIDENCE_BUCKETS
    }
    for confidence, is_relevant in pairs:
        if confidence not in buckets:
            continue
        buckets[confidence]["total"] += 1
        if is_relevant:
            buckets[confidence]["relevant"] += 1

    result: Dict[str, Any] = {}
    precisions: List[float] = []
    for bucket in _CONFIDENCE_BUCKETS:
        total = buckets[bucket]["total"]
        relevant = buckets[bucket]["relevant"]
        precision = (relevant / total) if total else None
        result[bucket] = {"relevant": relevant, "total": total, "precision": precision}
        if precision is not None:
            precisions.append(precision)

    result["monotonic"] = all(
        precisions[i] >= precisions[i + 1] for i in range(len(precisions) - 1)
    )
    return result


def _load_labeled_queries(queries_path: Path) -> Tuple[Path, List[LabeledQuery]]:
    """Parse the eval_queries.json schema; resolve ``corpus_root`` relative
    to *queries_path*'s own directory."""
    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    corpus_root = queries_path.parent / payload["corpus_root"]
    queries: List[LabeledQuery] = []
    for entry in payload["queries"]:
        relevant = [
            RelevantSpan(
                source_path=span["source_path"],
                start_line=span.get("start_line"),
                end_line=span.get("end_line"),
            )
            for span in entry["relevant"]
        ]
        queries.append(
            LabeledQuery(
                id=entry["id"], query=entry["query"], category=entry["category"],
                relevant=relevant,
            )
        )
    return corpus_root, queries


def _make_eval_retriever(
    name: str, tmp_dir_holder: List[Optional[str]], params: Optional[Dict[str, Any]] = None
) -> Retriever:
    """Instantiate *name*; ``pi-serini`` gets a durable per-eval-run Lucene
    dir (mirrors ``cli._make_retriever``) since it can't index without one.

    The Lucene dir is created lazily, only once pi-serini is actually about
    to be built, and stashed in *tmp_dir_holder* (a 1-element list used as an
    out-param) so the caller can clean it up regardless of which strategies
    ran.
    """
    if name == "pi-serini":
        if tmp_dir_holder[0] is None:
            tmp_dir_holder[0] = tempfile.mkdtemp(prefix="retrieval_eval_")
        ctor_kwargs = resolve_ctor_kwargs("pi-serini", PiSeriniRetriever, params)
        return PiSeriniRetriever(
            index_path=Path(tmp_dir_holder[0]) / "pi-serini-lucene", **ctor_kwargs
        )
    return build_retriever(name, params)


def _documents_for(root: Path, name: str, policy=None, loader_kw: Optional[Dict[str, Any]] = None):
    if name == "treesitter":
        return load_ast_chunk_documents(root, policy=policy, **(loader_kw or {}))
    return load_chunk_documents(root, policy=policy, **(loader_kw or {}))


def _build_retrievers(
    root: Path,
    params: Optional[Dict[str, Any]] = None,
    policy=None,
    loader_kw: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Retriever], List[Dict[str, str]], Dict[str, float], Optional[str]]:
    """Build every strategy in ``_STRATEGIES`` in-memory (no persistence),
    skipping (not failing) any whose optional extras are missing.

    Returns the pi-serini Lucene tmp dir (or ``None`` if pi-serini was never
    built) alongside the usual results, so the caller can remove it once the
    eval run is done.
    """
    retrievers: Dict[str, Retriever] = {}
    skipped: List[Dict[str, str]] = []
    build_s: Dict[str, float] = {}
    tmp_dir_holder: List[Optional[str]] = [None]
    for name in _STRATEGIES:
        try:
            retriever = _make_eval_retriever(name, tmp_dir_holder, params)
            documents = _documents_for(root, name, policy, loader_kw)
            start = time.perf_counter()
            retriever.index(documents)
            build_s[name] = time.perf_counter() - start
            retrievers[name] = retriever
        except RuntimeError as exc:
            skipped.append({"name": name, "reason": str(exc).splitlines()[0]})
    return retrievers, skipped, build_s, tmp_dir_holder[0]


def _timed_search(
    retriever: Retriever, query: str, pool: int, warm_runs: int
) -> Tuple[List[SearchHit], float, float]:
    """Search once (cold) then ``warm_runs`` more times (warm); returns
    ``(hits, cold_seconds, warm_median_seconds)``."""
    start = time.perf_counter()
    hits = retriever.search_detailed(query, pool)
    cold = time.perf_counter() - start

    warm_samples = []
    for _ in range(max(warm_runs, 0)):
        start = time.perf_counter()
        retriever.search_detailed(query, pool)
        warm_samples.append(time.perf_counter() - start)
    warm = statistics.median(warm_samples) if warm_samples else 0.0
    return hits, cold, warm


def _timed_consolidate(
    per_retriever_hits: Dict[str, List[SearchHit]], warm_runs: int
) -> Tuple[List[Any], float, float]:
    """Like ``_timed_search`` but for ``consolidate()`` itself — cold is the
    first fusion call, warm is the median of ``warm_runs`` repeats."""
    start = time.perf_counter()
    consolidated = consolidate(per_retriever_hits)
    cold = time.perf_counter() - start

    warm_samples = []
    for _ in range(max(warm_runs, 0)):
        start = time.perf_counter()
        consolidate(per_retriever_hits)
        warm_samples.append(time.perf_counter() - start)
    warm = statistics.median(warm_samples) if warm_samples else 0.0
    return consolidated, cold, warm


def _eval_single_query(
    query: LabeledQuery, retrievers: Dict[str, Retriever], k: int, warm_runs: int
) -> QueryEval:
    # Not capped to a corpus size: the consolidation quadratic span-merge
    # stays cheap enough at this depth for the eval harness's per-query
    # corpora (see the same tradeoff note in cli._query_all).
    pool = candidate_pool(k)
    per_retriever_hits: Dict[str, List[SearchHit]] = {}
    query_eval = QueryEval(query_id=query.id, category=query.category)

    for name, retriever in retrievers.items():
        hits, cold, warm = _timed_search(retriever, query.query, pool, warm_runs)
        per_retriever_hits[name] = hits
        query_eval.per_retriever[name] = RetrieverEval(
            name=name,
            recall_at_k=recall_at_k(hits, query.relevant, k),
            ndcg_at_k=ndcg_at_k(hits, query.relevant, k),
            cold_search_s=cold,
            warm_search_s=warm,
        )

    consolidated_hits, cold, warm = _timed_consolidate(per_retriever_hits, warm_runs)
    query_eval.per_retriever["consolidated"] = RetrieverEval(
        name="consolidated",
        recall_at_k=recall_at_k(consolidated_hits, query.relevant, k),
        ndcg_at_k=ndcg_at_k(consolidated_hits, query.relevant, k),
        cold_search_s=cold,
        warm_search_s=warm,
    )
    query_eval.confidence_hits = [
        (hit.confidence, any(span_overlaps(hit, gold) for gold in query.relevant))
        for hit in consolidated_hits[:k]
    ]
    return query_eval


def _aggregate(query_evals: List[QueryEval]) -> Dict[str, Dict[str, float]]:
    """Mean of each metric per retriever, over only the queries where that
    retriever actually ran (i.e. wasn't skipped)."""
    samples: Dict[str, Dict[str, List[float]]] = {}
    for query_eval in query_evals:
        for name, retriever_eval in query_eval.per_retriever.items():
            bucket = samples.setdefault(
                name,
                {"recall_at_k": [], "ndcg_at_k": [], "cold_search_s": [], "warm_search_s": []},
            )
            bucket["recall_at_k"].append(retriever_eval.recall_at_k)
            bucket["ndcg_at_k"].append(retriever_eval.ndcg_at_k)
            bucket["cold_search_s"].append(retriever_eval.cold_search_s)
            bucket["warm_search_s"].append(retriever_eval.warm_search_s)

    return {
        name: {metric: statistics.mean(values) for metric, values in metrics.items() if values}
        for name, metrics in samples.items()
    }


def run_eval(
    queries_path: _PathLike,
    root: Optional[_PathLike] = None,
    k: int = 5,
    warm_runs: int = 5,
    params: Optional[Dict[str, Any]] = None,
    loader_kw: Optional[Dict[str, Any]] = None,
) -> EvalReport:
    """Run the full eval harness over the labeled query set at *queries_path*.

    Builds every strategy in ``_STRATEGIES`` in-memory over the corpus (the
    query set's own ``corpus_root``, unless *root* overrides it), searches
    each surviving strategy per query, consolidates their hits, scores
    recall@k/nDCG@k for every retriever (including ``"consolidated"``),
    profiles cold/warm search latency, and tallies confidence-bucket
    precision over the consolidated top-k hits across every query.

    *params* (default ``None``, reproducing today's static-default behavior)
    is threaded into each retriever's construction (filtered to its own
    ``ACCEPTS``) and into chunking via a ``ChunkingPolicy`` built from its
    chunking keys — see ``retrieval.autotune.resolve_params``. *loader_kw*
    (default ``None`` -> ``{}``) is forwarded to ``discover_files`` via each
    strategy's document loader.
    """
    queries_path = Path(queries_path)
    default_root, labeled_queries = _load_labeled_queries(queries_path)
    corpus_root = Path(root) if root is not None else default_root
    policy = ChunkingPolicy.from_dict(params) if params else None

    retrievers, skipped, build_s, tmp_dir = _build_retrievers(
        corpus_root, params, policy, loader_kw
    )
    try:
        query_evals = [
            _eval_single_query(query, retrievers, k, warm_runs) for query in labeled_queries
        ]
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    all_confidence_pairs: List[Tuple[str, bool]] = []
    for query_eval in query_evals:
        all_confidence_pairs.extend(query_eval.confidence_hits)

    return EvalReport(
        k=k,
        queries=query_evals,
        aggregate=_aggregate(query_evals),
        confidence_validity=confidence_validity(all_confidence_pairs),
        skipped=skipped,
        build_s=build_s,
    )


def report_to_dict(report: EvalReport) -> Dict[str, Any]:
    """JSON-safe serialization of an ``EvalReport``."""
    return {
        "k": report.k,
        "queries": [
            {
                "query_id": query_eval.query_id,
                "category": query_eval.category,
                "per_retriever": {
                    name: {
                        "recall_at_k": retriever_eval.recall_at_k,
                        "ndcg_at_k": retriever_eval.ndcg_at_k,
                        "cold_search_s": retriever_eval.cold_search_s,
                        "warm_search_s": retriever_eval.warm_search_s,
                    }
                    for name, retriever_eval in query_eval.per_retriever.items()
                },
                "confidence_hits": [list(pair) for pair in query_eval.confidence_hits],
            }
            for query_eval in report.queries
        ],
        "aggregate": report.aggregate,
        "confidence_validity": report.confidence_validity,
        "skipped": report.skipped,
        "build_s": report.build_s,
    }


def format_report_text(report: EvalReport) -> str:
    """Human-readable text rendering of an ``EvalReport``."""
    lines = [f"Eval report (k={report.k}, queries={len(report.queries)})", ""]

    if report.skipped:
        lines.append("Skipped:")
        for note in report.skipped:
            lines.append(f"  {note['name']}: {note['reason']}")
        lines.append("")

    lines.append("Aggregate per retriever (mean over queries where it ran):")
    for name in sorted(report.aggregate):
        metrics = report.aggregate[name]
        lines.append(
            f"  {name:<14} recall@k={metrics['recall_at_k']:.3f}  "
            f"ndcg@k={metrics['ndcg_at_k']:.3f}  "
            f"cold={metrics['cold_search_s'] * 1000:.2f}ms  "
            f"warm={metrics['warm_search_s'] * 1000:.2f}ms"
        )
    lines.append("")

    lines.append("Confidence-signal validity (consolidated top-k hits, all queries):")
    for bucket in _CONFIDENCE_BUCKETS:
        info = report.confidence_validity.get(bucket, {})
        precision = info.get("precision")
        precision_str = f"{precision:.3f}" if precision is not None else "n/a"
        lines.append(
            f"  {bucket:<6} precision={precision_str}  "
            f"relevant={info.get('relevant', 0)}/{info.get('total', 0)}"
        )
    monotonic = report.confidence_validity.get("monotonic")
    lines.append(f"  monotonic (high >= medium >= low): {monotonic}")

    return "\n".join(lines)
