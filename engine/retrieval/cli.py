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
from retrieval.consolidation import ConsolidatedHit, consolidate
from retrieval.document import SearchHit
from retrieval.eval import format_report_text, report_to_dict, run_eval
from retrieval.persistence import (
    cached_retrievers,
    compute_fingerprint,
    index_dir,
    is_stale,
    load_index,
    save_index,
)
from retrieval.project_loader import load_ast_chunk_documents, load_chunk_documents
from retrieval.retrievers import PiSeriniRetriever, Retriever, build_retriever

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
    query_parser.add_argument("--top-k", type=int, default=5, help="number of results (default: 5)")
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

    return parser


def _make_retriever(root: Path, retriever_name: str) -> Retriever:
    """Instantiate *retriever_name*; pi-serini gets a durable Lucene dir
    inside the per-project cache so its index survives for ``query``."""
    if retriever_name == "pi-serini":
        return PiSeriniRetriever(index_path=index_dir(root) / "lucene")
    return build_retriever(retriever_name)


def _documents_for(root: Path, retriever_name: str):
    """Return the chunk-granularity Documents to index for *retriever_name*.

    ``treesitter`` uses AST-boundary chunks (carrying a ``context``
    breadcrumb); every other retriever uses the line-based chunker.
    """
    if retriever_name == "treesitter":
        return load_ast_chunk_documents(root)
    return load_chunk_documents(root)


def _build_and_save(root: Path, retriever_name: str) -> Retriever:
    """Build a fresh retriever over *root* and persist it; return the retriever."""
    fingerprint = compute_fingerprint(root)
    retriever = _make_retriever(root, retriever_name)
    retriever.index(_documents_for(root, retriever_name))
    save_index(retriever, root, fingerprint, retriever_name, __version__)
    return retriever


def _up_to_date_message(root: Path, retriever_name: str) -> Optional[str]:
    """Return a "fast path" message if a fresh, non-stale cache exists for
    *root* + *retriever_name*; ``None`` if there's no cache or it's stale
    (a rebuild is needed).
    """
    cached = load_index(root, retriever_name)
    if cached is None:
        return None
    _retriever, meta = cached
    if is_stale(root, meta):
        return None
    return (
        f"{retriever_name} index up to date -> {index_dir(root)}  (use --force to rebuild)"
    )


def _index_all(root: Path, force: bool) -> int:
    """Build every strategy in ``_DEFAULT_INDEX_SET``, skipping (not failing)
    any whose optional extras aren't installed. Only a failure to build the
    always-available ``lexical`` strategy is treated as a hard failure.
    """
    lexical_failed = False
    for name in _DEFAULT_INDEX_SET:
        if not force and _up_to_date_message(root, name) is not None:
            print(f"{name}: up to date (use --force to rebuild)")
            continue
        try:
            retriever = _build_and_save(root, name)
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
    if args.retriever == "all":
        return _index_all(root, args.force)
    if not args.force:
        message = _up_to_date_message(root, args.retriever)
        if message is not None:
            print(message)
            return 0
    fingerprint = compute_fingerprint(root)
    retriever = _make_retriever(root, args.retriever)
    retriever.index(_documents_for(root, args.retriever))
    saved_dir = save_index(retriever, root, fingerprint, args.retriever, __version__)
    chunk_count = len(retriever.to_dict()["docids"])
    print(f"indexed {chunk_count} chunks -> {saved_dir}  fingerprint={fingerprint[:12]}")
    return 0


def _load_or_rebuild(root: Path, retriever_name: str, stale_ok: bool) -> Retriever:
    """Load the cached retriever for *root*, rebuilding it if missing/stale."""
    cached = load_index(root, retriever_name)
    if cached is None:
        return _build_and_save(root, retriever_name)
    retriever, meta = cached
    if is_stale(root, meta) and not stale_ok:
        return _build_and_save(root, retriever_name)
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


def _query_all(root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Load/rebuild every strategy in ``_DEFAULT_INDEX_SET`` (skipping any
    whose extras are missing), search each, and consolidate the results.

    Returns a dict with ``retrievers`` (names successfully consolidated),
    ``skipped`` (``[{name, reason}]``), and ``results`` (``ConsolidatedHit``
    list truncated to ``args.top_k``).
    """
    pool = max(args.top_k * 3, 10)
    per_retriever_hits: Dict[str, List[SearchHit]] = {}
    skipped: List[Dict[str, str]] = []
    for name in _DEFAULT_INDEX_SET:
        try:
            retriever = _load_or_rebuild(root, name, args.stale_ok)
            per_retriever_hits[name] = retriever.search_detailed(args.query, pool)
        except RuntimeError as exc:
            skipped.append({"name": name, "reason": str(exc).splitlines()[0]})

    weights = _parse_weights(args.weights)
    consolidated = consolidate(per_retriever_hits, weights=weights)[: args.top_k]
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
    hits: List[SearchHit] = retriever.search_detailed(args.query, args.top_k)
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


def _cmd_eval(args: argparse.Namespace) -> int:
    report = run_eval(args.queries, root=args.root, k=args.k, warm_runs=args.warm_runs)
    envelope = report_to_dict(report)
    _write_output(args.output, envelope)
    if args.json:
        print(json.dumps(envelope))
    else:
        print(format_report_text(report))
    # Mirror the query/index convention: exit 0 as long as the always-
    # available `lexical` strategy produced at least one query result.
    return 1 if "lexical" not in report.aggregate else 0


_COMMANDS = {
    "index": _cmd_index,
    "query": _cmd_query,
    "stats": _cmd_stats,
    "eval": _cmd_eval,
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
