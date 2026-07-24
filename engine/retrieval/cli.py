"""``retrieval`` console-script CLI: setup/index/query/stats/eval over
persisted, on-disk retriever indexes.

Thin argparse dispatcher over ``retrieval.persistence`` + ``retrieval.retrievers``
— every subcommand resolves a project root (``--root`` -> ``RETRIEVAL_ROOT`` env
-> cwd), then either builds+saves a fresh index (``index``), loads/reindexes and
searches it (``query``), reports on the caches (``stats``), or runs the
labeled-query eval harness in-memory with no persistence (``eval``). ``query``'s
default (``--retriever all``) consolidates every available strategy's
ranking into one deduplicated, explainable list via
``retrieval.consolidation.consolidate``; pass an explicit ``--retriever
<name>`` to query exactly one strategy instead, with output byte-identical
to before consolidated mode existed. The always-available ``lexical``
retriever needs no optional extras; ``lexical+ctx`` needs whatever the
contextualizer needs, ``turbovec``/``hybrid`` need the turbovec + local
extras, ``pi-serini`` needs the pyserini extra plus Java 21, and
``treesitter`` needs the treesitter extra (each raises a guidance
RuntimeError when its extras are missing).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval import __version__
from retrieval.autotune import (
    CorpusSignals,
    collect_signals,
    resolve_params,
    resolve_top_k,
    thin_chunk_warning,
)
from retrieval.chunker import ChunkingPolicy
from retrieval.consolidation import ConsolidatedHit, consolidate
from retrieval.document import SearchHit
from retrieval.eval import format_report_text, report_to_dict, run_eval
from retrieval.fusion import candidate_pool
from retrieval.persistence import (
    cached_retrievers,
    compute_fingerprint,
    index_dir,
    is_stale,
    load_index,
    save_index,
)
from retrieval.project_loader import load_ast_chunk_documents, load_chunk_documents
from retrieval.retrievers import (
    PiSeriniRetriever,
    Retriever,
    build_retriever,
    resolve_ctor_kwargs,
)

_RETRIEVER_CHOICES = (
    "lexical", "lexical+ctx", "turbovec", "pi-serini", "hybrid", "treesitter",
)
# lexical+ctx deliberately excluded: shares the lexical cache slot and costs LLM tokens.
_DEFAULT_INDEX_SET = ("lexical", "turbovec", "pi-serini", "hybrid", "treesitter")
_INDEX_RETRIEVER_CHOICES = _RETRIEVER_CHOICES + ("all",)
_QUERY_RETRIEVER_CHOICES = _RETRIEVER_CHOICES + ("all",)


def _resolve_root(root_arg: Optional[str]) -> Path:
    """Resolve the project root: ``--root`` -> ``RETRIEVAL_ROOT`` env -> cwd."""
    if root_arg:
        return Path(root_arg).resolve()
    env_root = os.environ.get("RETRIEVAL_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="build and persist an index for a project root"
    )
    index_parser.add_argument(
        "--root", help="project root to index (default: RETRIEVAL_ROOT or cwd)"
    )
    index_parser.add_argument(
        "--retriever", default="all", choices=_INDEX_RETRIEVER_CHOICES,
        help="retriever to build, or 'all' for every default strategy (default: all)",
    )
    index_parser.add_argument(
        "--force", action="store_true",
        help="rebuild even if a fresh (non-stale) cache already exists "
        "(default: skip the rebuild and report up-to-date)",
    )
    index_parser.add_argument(
        "--auto", action="store_true",
        help="auto-derive hyperparameters from corpus signals (see the "
        "'tune' subcommand); explicit flags below still override",
    )
    index_parser.add_argument(
        "--target-chars", type=int, default=None,
        help="chunk-size bucket for files not matched by --config-chars/--code-chars",
    )
    index_parser.add_argument("--config-chars", type=int, default=None)
    index_parser.add_argument("--code-chars", type=int, default=None)
    index_parser.add_argument("--ast-max-chars", type=int, default=None)
    index_parser.add_argument("--bm25-k1", type=float, default=None)
    index_parser.add_argument("--bm25-b", type=float, default=None)
    index_parser.add_argument("--tokenizer", choices=["plain", "code"], default=None)
    index_parser.add_argument("--bit-width", type=int, default=None)
    index_parser.add_argument(
        "--embed-model", default=None, help="turbovec/hybrid embedding model name"
    )
    index_parser.add_argument("--lucene-k1", type=float, default=None)
    index_parser.add_argument("--lucene-b", type=float, default=None)

    query_parser = subparsers.add_parser(
        "query", help="search the persisted index for a project root"
    )
    query_parser.add_argument("query", help="query text")
    query_parser.add_argument(
        "--root", help="project root to search (default: RETRIEVAL_ROOT or cwd)"
    )
    query_parser.add_argument(
        "--retriever", default="all", choices=_QUERY_RETRIEVER_CHOICES,
        help="retriever to query, or 'all' (default) to consolidate every "
        "available strategy into a single deduplicated, explainable ranking",
    )
    query_parser.add_argument(
        "--top-k", type=int, default=None,
        help="number of results (default: 5, or an auto-derived estimate with "
        "--auto and a persisted corpus_stats block)",
    )
    query_parser.add_argument(
        "--auto", action="store_true",
        help="resolve --top-k from the index's persisted corpus_stats (see "
        "'tune'); ignored when --top-k is given explicitly. Never re-derives "
        "index-time hyperparameters (tokenizer, etc.) — those come from the "
        "cache's own metadata.",
    )
    query_parser.add_argument(
        "--pool", type=int, default=None,
        help="consolidated-mode only: override the per-retriever candidate-pool "
        "depth (default: derived from --top-k via retrieval.fusion.candidate_pool)",
    )
    query_parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of one docid per line"
    )
    query_parser.add_argument(
        "--stale-ok", action="store_true",
        help="search the cached index even if it's stale, instead of auto-reindexing",
    )
    query_parser.add_argument(
        "--weights", default=None,
        help="consolidated-mode only: comma-separated 'name:weight' pairs "
        "(e.g. 'lexical:1.5,turbovec:0.5') overriding a retriever's RRF "
        "weight; unlisted retrievers default to 1.0",
    )
    query_parser.add_argument(
        "--output", default=None,
        help="consolidated-mode only: also write the JSON envelope to this path",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="report on the persisted index for a project root"
    )
    stats_parser.add_argument(
        "--root", help="project root to report on (default: RETRIEVAL_ROOT or cwd)"
    )

    tune_parser = subparsers.add_parser(
        "tune",
        help="report auto-derived hyperparameters for a project root's corpus "
        "signals without writing anything (see 'index --auto')",
    )
    tune_parser.add_argument(
        "--root", help="project root to analyze (default: RETRIEVAL_ROOT or cwd)"
    )
    tune_parser.add_argument(
        "--json", action="store_true", help="emit a JSON report instead of text"
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="run the labeled-query eval harness (recall@k/nDCG@k, confidence "
        "validity, cold/warm latency) in-memory, no persistence",
    )
    eval_parser.add_argument(
        "--queries", required=True, help="path to a labeled eval_queries.json file"
    )
    eval_parser.add_argument(
        "--root",
        help="corpus root to eval against (default: the queries file's own "
        "'corpus_root', resolved relative to the queries file)",
    )
    eval_parser.add_argument(
        "--k", type=int, default=5, help="recall@k / nDCG@k cutoff (default: 5)"
    )
    eval_parser.add_argument(
        "--warm-runs", type=int, default=5,
        help="number of extra warm search repeats per query (default: 5)",
    )
    eval_parser.add_argument(
        "--json", action="store_true", help="emit a JSON report instead of text"
    )
    eval_parser.add_argument(
        "--output", default=None, help="also write the report to this path"
    )
    eval_parser.add_argument(
        "--auto", action="store_true",
        help="auto-derive hyperparameters from the eval corpus's signals; "
        "recorded in the report as 'auto_params'",
    )

    return parser


def _make_retriever(
    root: Path, retriever_name: str, params: Optional[Dict[str, Any]] = None
) -> Retriever:
    """Instantiate *retriever_name*; pi-serini gets a durable Lucene dir
    inside the per-project cache so its index survives for ``query``."""
    if retriever_name == "pi-serini":
        ctor_kwargs = resolve_ctor_kwargs("pi-serini", PiSeriniRetriever, params)
        return PiSeriniRetriever(index_path=index_dir(root) / "lucene", **ctor_kwargs)
    return build_retriever(retriever_name, params)


def _documents_for(
    root: Path, retriever_name: str, policy: Optional[ChunkingPolicy] = None
):
    """Return the chunk-granularity Documents to index for *retriever_name*.

    ``treesitter`` uses AST-boundary chunks (carrying a ``context``
    breadcrumb); every other retriever uses the line-based chunker. *policy*
    (default ``None`` -> ``DEFAULT_POLICY``, today's flat 400-char chunking)
    picks each file's chunk-size bucket by suffix.
    """
    if retriever_name == "treesitter":
        return load_ast_chunk_documents(root, policy=policy)
    return load_chunk_documents(root, policy=policy)


#: --index-flag -> params-dict key, for the hyperparameter overrides an
#: explicit CLI flag can layer on top of --auto (or a bare static default
#: when --auto isn't given). See ``_resolve_index_params``.
_INDEX_OVERRIDE_ARGS = {
    "target_chars": "default_chars",
    "config_chars": "config_chars",
    "code_chars": "code_chars",
    "ast_max_chars": "ast_max_chars",
    "bm25_k1": "bm25_k1",
    "bm25_b": "bm25_b",
    "tokenizer": "tokenizer",
    "bit_width": "bit_width",
    "embed_model": "model_name",
    "lucene_k1": "lucene_k1",
    "lucene_b": "lucene_b",
}


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect the explicit hyperparameter-flag overrides present on *args*
    (any flag missing from the namespace, e.g. on the ``eval`` subparser
    which has no per-hyperparameter flags, is simply skipped)."""
    return {
        params_key: getattr(args, arg_name)
        for arg_name, params_key in _INDEX_OVERRIDE_ARGS.items()
        if hasattr(args, arg_name)
    }


def _resolve_index_params(
    root: Path, args: argparse.Namespace
) -> "tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[ChunkingPolicy]]":
    """Resolve ``(params, corpus_stats, policy)`` for ``index``/``eval``
    with precedence explicit flag > auto-derived > static default.

    Returns ``(None, None, None)`` — reproducing today's behavior exactly,
    with no ``hyperparams``/``corpus_stats`` meta block written — when
    neither ``--auto`` nor any hyperparameter flag was given.
    """
    overrides = _overrides_from_args(args)
    if not args.auto and not any(v is not None for v in overrides.values()):
        return None, None, None

    corpus_stats: Optional[Dict[str, Any]] = None
    if args.auto:
        signals = collect_signals(root)
        params = resolve_params(signals, overrides)
        corpus_stats = {
            "n_chunks": signals.n_chunks,
            "mean_chunk_tokens": signals.mean_chunk_tokens,
            "chunk_token_cv": signals.chunk_token_cv,
            "auto": True,
        }
    else:
        # No --auto: explicit flags layered directly over static defaults
        # (an empty-corpus CorpusSignals makes every _decide_* fall back).
        params = resolve_params(CorpusSignals(n_files=0), overrides)
    policy = ChunkingPolicy.from_dict(params)
    return params, corpus_stats, policy


def _build_and_save(
    root: Path,
    retriever_name: str,
    params: Optional[Dict[str, Any]] = None,
    corpus_stats: Optional[Dict[str, Any]] = None,
    policy: Optional[ChunkingPolicy] = None,
) -> Retriever:
    """Build a fresh retriever over *root* and persist it; return the retriever."""
    fingerprint = compute_fingerprint(root)
    retriever = _make_retriever(root, retriever_name, params)
    retriever.index(_documents_for(root, retriever_name, policy))
    save_index(
        retriever, root, fingerprint, retriever_name, __version__,
        params=params, corpus_stats=corpus_stats,
    )
    return retriever


def _up_to_date_message(
    root: Path, retriever_name: str, params: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Return a "fast path" message if a fresh, non-stale cache exists for
    *root* + *retriever_name*; ``None`` if there's no cache or it's stale
    (a rebuild is needed).
    """
    cached = load_index(root, retriever_name)
    if cached is None:
        return None
    _retriever, meta = cached
    if is_stale(root, meta, params=params):
        return None
    return (
        f"{retriever_name} index up to date -> {index_dir(root)}  (use --force to rebuild)"
    )


def _index_all(
    root: Path,
    force: bool,
    params: Optional[Dict[str, Any]] = None,
    corpus_stats: Optional[Dict[str, Any]] = None,
    policy: Optional[ChunkingPolicy] = None,
) -> int:
    """Build every strategy in ``_DEFAULT_INDEX_SET``, skipping (not failing)
    any whose optional extras aren't installed. Only a failure to build the
    always-available ``lexical`` strategy is treated as a hard failure.

    When *params* is ``None`` (no ``--auto``, no explicit hyperparameter
    flag), each strategy's own previously recorded hyperparameters are
    recovered from its existing meta (see ``_params_for_rebuild``) — so a
    routine ``retrieval index`` refresh (with or without ``--force``) never
    silently reverts a strategy's recorded decisions to static defaults.
    Explicit flags/``--auto`` (non-``None`` *params*) always win and are
    applied identically to every strategy, as before.
    """
    lexical_failed = False
    for name in _DEFAULT_INDEX_SET:
        if not force and _up_to_date_message(root, name, params) is not None:
            print(f"{name}: up to date (use --force to rebuild)")
            continue
        build_params, build_corpus_stats, build_policy = params, corpus_stats, policy
        if build_params is None:
            build_params, build_corpus_stats, build_policy = _params_for_rebuild(root, name, None)
        try:
            retriever = _build_and_save(root, name, build_params, build_corpus_stats, build_policy)
            chunk_count = len(retriever.to_dict()["docids"])
            print(f"{name}: indexed {chunk_count} chunks")
        except RuntimeError as exc:
            print(f"{name}: skipped ({str(exc).splitlines()[0]})")
            if name == "lexical":
                lexical_failed = True
    print(f"-> {index_dir(root)}")
    return 1 if lexical_failed else 0


def _cmd_index(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    params, corpus_stats, policy = _resolve_index_params(root, args)
    if args.retriever == "all":
        return _index_all(root, args.force, params, corpus_stats, policy)
    if not args.force:
        message = _up_to_date_message(root, args.retriever, params)
        if message is not None:
            print(message)
            return 0
    # Nothing explicit was given (no --auto, no hyperparameter flag): a
    # routine refresh (with or without --force) must not silently revert
    # this retriever's own previously recorded hyperparameters to static
    # defaults — recover them from its existing meta, if any.
    if params is None:
        params, corpus_stats, policy = _params_for_rebuild(root, args.retriever, None)
    fingerprint = compute_fingerprint(root)
    retriever = _make_retriever(root, args.retriever, params)
    retriever.index(_documents_for(root, args.retriever, policy))
    saved_dir = save_index(
        retriever, root, fingerprint, args.retriever, __version__,
        params=params, corpus_stats=corpus_stats,
    )
    chunk_count = len(retriever.to_dict()["docids"])
    print(f"indexed {chunk_count} chunks -> {saved_dir}  fingerprint={fingerprint[:12]}")
    return 0


def _peek_meta(root: Path, retriever_name: str) -> Optional[Dict[str, Any]]:
    """Best-effort meta lookup that survives a corrupt/incompatible data
    file: ``cached_retrievers`` only ever reads meta.json (never the
    heavier data file ``load_index`` also requires and can fail on), so it
    can still recover a previous run's recorded hyperparameters even when
    ``load_index`` itself returns ``None``."""
    cached = cached_retrievers(root)
    if retriever_name in cached:
        return cached[retriever_name]
    if retriever_name in ("lexical", "lexical+ctx"):
        # Shared cache slot: whichever of the two built it last is keyed
        # under its own name in cached_retrievers, not necessarily the one
        # being asked about here.
        return cached.get("lexical") or cached.get("lexical+ctx")
    return None


def _params_for_rebuild(
    root: Path,
    retriever_name: str,
    params: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> "tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[ChunkingPolicy]]":
    """Resolve ``(params, corpus_stats, policy)`` to rebuild with.

    Explicit *params* (the caller asked for specific hyperparameters) always
    wins. Otherwise — the common case: a plain ``query``/``_load_or_rebuild``
    call with no --auto/hyperparameter flags — recover the index's own
    previously recorded ``meta["hyperparams"]``/``meta["corpus_stats"]``
    (from *meta*, or a best-effort ``_peek_meta`` lookup when *meta* wasn't
    already loaded, e.g. the ``load_index`` returned ``None`` path) so a
    query-triggered rebuild never silently reverts to static defaults.
    Falls back to ``(None, None, None)`` only when no meta is readable at
    all (nothing to recover from).
    """
    if params is not None:
        return params, None, ChunkingPolicy.from_dict(params)
    if meta is None:
        meta = _peek_meta(root, retriever_name)
    if meta is None:
        return None, None, None
    hyperparams = meta.get("hyperparams")
    policy = ChunkingPolicy.from_dict(hyperparams) if hyperparams else None
    return hyperparams, meta.get("corpus_stats"), policy


def _load_or_rebuild(
    root: Path, retriever_name: str, stale_ok: bool, params: Optional[Dict[str, Any]] = None
) -> Retriever:
    """Load the cached retriever for *root*, rebuilding it if missing/stale.

    A rebuild (whether triggered by no cache at all or by a stale
    fingerprint/hyperparams) preserves the index's own previously recorded
    hyperparameters — see ``_params_for_rebuild`` — rather than silently
    reverting to static defaults just because the caller didn't pass
    explicit *params*.
    """
    cached = load_index(root, retriever_name)
    if cached is None:
        rebuild_params, rebuild_corpus_stats, rebuild_policy = _params_for_rebuild(
            root, retriever_name, params
        )
        return _build_and_save(
            root, retriever_name, rebuild_params, rebuild_corpus_stats, rebuild_policy
        )
    retriever, meta = cached
    if is_stale(root, meta, params=params) and not stale_ok:
        rebuild_params, rebuild_corpus_stats, rebuild_policy = _params_for_rebuild(
            root, retriever_name, params, meta
        )
        return _build_and_save(
            root, retriever_name, rebuild_params, rebuild_corpus_stats, rebuild_policy
        )
    return retriever


def _hit_to_json(hit: SearchHit) -> Dict[str, Any]:
    return {
        "docid": hit.docid,
        "path": hit.source_path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "rank": hit.rank,
        "context": hit.context,
    }


def _consolidated_hit_to_json(hit: ConsolidatedHit) -> Dict[str, Any]:
    return {
        "docid": hit.docid,
        "path": hit.source_path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "rank": hit.rank,
        "context": hit.context,
        "score": hit.score,
        "provenance": hit.provenance,
        "agreement": hit.agreement,
        "confidence": hit.confidence,
        "contributors": hit.contributors,
    }


def _parse_weights(weights_arg: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse ``--weights "name:w,name:w"`` into ``{name: float(w)}``; ``None``
    if no ``--weights`` was given."""
    if not weights_arg:
        return None
    weights: Dict[str, float] = {}
    for pair in weights_arg.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition(":")
        weights[name.strip()] = float(value.strip())
    return weights


def _resolve_query_top_k(args: argparse.Namespace, meta: Optional[Dict[str, Any]]) -> int:
    """``--top-k`` precedence: explicit flag > (``--auto`` + the index's
    persisted ``corpus_stats``, via ``retrieval.autotune.resolve_top_k``) >
    the static default of 5. Never re-derives index-time hyperparameters —
    only reads back what indexing already recorded."""
    if args.top_k is not None:
        return args.top_k
    if args.auto:
        return resolve_top_k((meta or {}).get("corpus_stats"))
    return 5


def _query_all(root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Load/rebuild every strategy in ``_DEFAULT_INDEX_SET`` (skipping any
    whose extras are missing), search each, and consolidate the results.

    Returns a dict with ``retrievers`` (names successfully consolidated),
    ``skipped`` (``[{name, reason}]``), and ``results`` (``ConsolidatedHit``
    list truncated to the resolved ``top_k``).
    """
    lexical_cached = load_index(root, "lexical")
    top_k = _resolve_query_top_k(args, lexical_cached[1] if lexical_cached else None)
    # Pool depth isn't capped to a corpus size here: each strategy has its
    # own indexed unit count, and consolidation itself (a quadratic
    # span-merge over the pooled hits) stays cheap enough at this depth to
    # not need one either.
    pool = args.pool if args.pool is not None else candidate_pool(top_k)
    per_retriever_hits: Dict[str, List[SearchHit]] = {}
    skipped: List[Dict[str, str]] = []
    for name in _DEFAULT_INDEX_SET:
        try:
            retriever = _load_or_rebuild(root, name, args.stale_ok)
            per_retriever_hits[name] = retriever.search_detailed(args.query, pool)
        except RuntimeError as exc:
            skipped.append({"name": name, "reason": str(exc).splitlines()[0]})

    weights = _parse_weights(args.weights)
    consolidated = consolidate(per_retriever_hits, weights=weights)[:top_k]
    return {
        "retrievers": sorted(per_retriever_hits),
        "skipped": skipped,
        "results": consolidated,
    }


def _print_consolidated_text(consolidated: Dict[str, Any]) -> None:
    for note in consolidated["skipped"]:
        print(f"{note['name']}: skipped ({note['reason']})", file=sys.stderr)
    for hit in consolidated["results"]:
        via = ",".join(hit.provenance)
        print(
            f"{hit.source_path}:{hit.start_line}-{hit.end_line}  "
            f"[score={hit.score:.4f} agree={hit.agreement}/{len(consolidated['retrievers'])} "
            f"conf={hit.confidence}  via {via}]  {hit.context}".rstrip()
        )


def _write_output(output_path: Optional[str], envelope: Dict[str, Any]) -> None:
    if not output_path:
        return
    Path(output_path).write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def _cmd_query_consolidated(args: argparse.Namespace, root: Path) -> int:
    consolidated = _query_all(root, args)
    envelope = {
        "query": args.query,
        "mode": "consolidated",
        "retrievers": consolidated["retrievers"],
        "skipped": consolidated["skipped"],
        "results": [_consolidated_hit_to_json(h) for h in consolidated["results"]],
    }
    _write_output(args.output, envelope)
    if args.json:
        print(json.dumps(envelope))
    else:
        _print_consolidated_text(consolidated)
    # Exit 0 as long as at least one retriever (any of them) got consolidated;
    # 1 only when even the always-available `lexical` strategy is unusable.
    return 1 if "lexical" not in consolidated["retrievers"] else 0


def _cmd_query(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    if args.retriever == "all":
        return _cmd_query_consolidated(args, root)
    retriever = _load_or_rebuild(root, args.retriever, args.stale_ok)
    cached = load_index(root, args.retriever)
    top_k = _resolve_query_top_k(args, cached[1] if cached else None)
    hits: List[SearchHit] = retriever.search_detailed(args.query, top_k)
    if args.json:
        print(json.dumps({"query": args.query, "results": [_hit_to_json(h) for h in hits]}))
    else:
        for hit in hits:
            print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}")
    return 0


def _print_stats(root: Path, retriever_name: str, meta: Dict[str, Any]) -> None:
    stale = is_stale(root, meta)
    print(f"retriever: {retriever_name}")
    print(f"root: {meta.get('corpus_root')}")
    print(f"chunks: {meta.get('doc_count')}")
    print(f"files: {meta.get('file_count')}")
    print(f"created: {meta.get('created_at')}")
    print(f"engine: {meta.get('engine_version')}")
    print(f"stale: {stale}")
    print(f"cache: {index_dir(root)}")
    if "hyperparams" in meta:
        print(f"hyperparams: {meta['hyperparams']}")
    if "corpus_stats" in meta:
        print(f"corpus_stats: {meta['corpus_stats']}")


def _cmd_stats(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    cached = cached_retrievers(root)
    if not cached:
        print(f"no cache for {root} (dir={index_dir(root)})")
        return 0
    for i, (retriever_name, meta) in enumerate(sorted(cached.items())):
        if i:
            print()
        _print_stats(root, retriever_name, meta)
    return 0


def _eval_corpus_root(args: argparse.Namespace) -> Path:
    """The corpus root ``run_eval`` will actually search against: an
    explicit ``--root``, else the queries file's own ``corpus_root``
    (mirrors ``eval._load_labeled_queries``'s resolution) — needed to
    collect ``--auto`` signals against the same root ``run_eval`` uses."""
    if args.root:
        return Path(args.root).resolve()
    queries_path = Path(args.queries)
    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    return (queries_path.parent / payload["corpus_root"]).resolve()


def _cmd_eval(args: argparse.Namespace) -> int:
    params = None
    if args.auto:
        signals = collect_signals(_eval_corpus_root(args))
        params = resolve_params(signals)
    report = run_eval(
        args.queries, root=args.root, k=args.k, warm_runs=args.warm_runs, params=params
    )
    envelope = report_to_dict(report)
    if params is not None:
        envelope["auto_params"] = params
    _write_output(args.output, envelope)
    if args.json:
        print(json.dumps(envelope))
    else:
        print(format_report_text(report))
        if params is not None:
            print(f"\nauto_params: {params}")
    # Mirror the query/index convention: exit 0 as long as the always-
    # available `lexical` strategy produced at least one query result.
    return 1 if "lexical" not in report.aggregate else 0


def _cmd_tune(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    signals = collect_signals(root)
    params = resolve_params(signals)
    warning = thin_chunk_warning(signals)
    if args.json:
        envelope = {
            "signals": {
                "n_files": signals.n_files,
                "ext_histogram": signals.ext_histogram,
                "total_bytes": signals.total_bytes,
                "code_bytes": signals.code_bytes,
                "code_fraction": signals.code_fraction,
                "median_code_lines": signals.median_code_lines,
                "n_chunks": signals.n_chunks,
                "mean_chunk_tokens": signals.mean_chunk_tokens,
                "chunk_token_cv": signals.chunk_token_cv,
                "p10_chunk_tokens": signals.p10_chunk_tokens,
                "p50_chunk_tokens": signals.p50_chunk_tokens,
                "p90_chunk_tokens": signals.p90_chunk_tokens,
            },
            "params": params,
            "warning": warning,
        }
        print(json.dumps(envelope))
        return 0
    print(f"signals: n_files={signals.n_files} code_fraction={signals.code_fraction:.3f}")
    print(
        f"         n_chunks={signals.n_chunks} "
        f"mean_chunk_tokens={signals.mean_chunk_tokens} "
        f"chunk_token_cv={signals.chunk_token_cv}"
    )
    print("proposed params (vs. static defaults):")
    for key in sorted(params):
        print(f"  {key}: {params[key]}")
    if warning:
        print(f"warning: {warning}")
    return 0


_COMMANDS = {
    "index": _cmd_index,
    "query": _cmd_query,
    "stats": _cmd_stats,
    "eval": _cmd_eval,
    "tune": _cmd_tune,
}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on a handled runtime error.

    Argument-parsing errors (missing/invalid flags) are handled by argparse
    itself and exit the process directly (``SystemExit``), before this
    function's error handling is reached.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except Exception as exc:  # noqa: BLE001 - CLI top-level error boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
