"""``retrieval`` console-script CLI: setup/index/query/stats over persisted,
on-disk retriever indexes.

Thin argparse dispatcher over ``retrieval.persistence`` + ``retrieval.retrievers``
— every subcommand resolves a project root (``--root`` -> ``RETRIEVAL_ROOT`` env
-> cwd), then either builds+saves a fresh index (``index``), loads/reindexes and
searches it (``query``), or reports on the caches (``stats``). The default
``lexical`` retriever needs no optional extras; ``lexical+ctx`` needs whatever
the contextualizer needs, ``turbovec``/``hybrid`` need the turbovec + local
extras, and ``pi-serini`` needs the pyserini extra plus Java 21 (each raises
a guidance RuntimeError when its extras are missing).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval import __version__
from retrieval.persistence import (
    cached_retrievers,
    compute_fingerprint,
    index_dir,
    is_stale,
    load_index,
    save_index,
)
from retrieval.project_loader import load_documents
from retrieval.retrievers import PiSeriniRetriever, Retriever, build_retriever

_RETRIEVER_CHOICES = ("lexical", "lexical+ctx", "turbovec", "pi-serini", "hybrid")
# lexical+ctx deliberately excluded: shares the lexical cache slot and costs LLM tokens.
_DEFAULT_INDEX_SET = ("lexical", "turbovec", "pi-serini", "hybrid")
_INDEX_RETRIEVER_CHOICES = _RETRIEVER_CHOICES + ("all",)


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
        "--retriever", default="lexical", choices=_RETRIEVER_CHOICES,
        help="retriever to use if the index needs (re)building (default: lexical)",
    )
    query_parser.add_argument("--top-k", type=int, default=5, help="number of results (default: 5)")
    query_parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of one docid per line"
    )
    query_parser.add_argument(
        "--stale-ok", action="store_true",
        help="search the cached index even if it's stale, instead of auto-reindexing",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="report on the persisted index for a project root"
    )
    stats_parser.add_argument(
        "--root", help="project root to report on (default: RETRIEVAL_ROOT or cwd)"
    )

    return parser


def _make_retriever(root: Path, retriever_name: str) -> Retriever:
    """Instantiate *retriever_name*; pi-serini gets a durable Lucene dir
    inside the per-project cache so its index survives for ``query``."""
    if retriever_name == "pi-serini":
        return PiSeriniRetriever(index_path=index_dir(root) / "lucene")
    return build_retriever(retriever_name)


def _build_and_save(root: Path, retriever_name: str) -> Retriever:
    """Build a fresh retriever over *root* and persist it; return the retriever."""
    fingerprint = compute_fingerprint(root)
    retriever = _make_retriever(root, retriever_name)
    retriever.index(load_documents(root))
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
            doc_count = len(retriever.to_dict()["docids"])
            print(f"{name}: indexed {doc_count} docs")
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
    retriever.index(load_documents(root))
    saved_dir = save_index(retriever, root, fingerprint, args.retriever, __version__)
    doc_count = len(retriever.to_dict()["docids"])
    print(f"indexed {doc_count} docs -> {saved_dir}  fingerprint={fingerprint[:12]}")
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


def _cmd_query(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    retriever = _load_or_rebuild(root, args.retriever, args.stale_ok)
    results: List[str] = retriever.search(args.query, args.top_k)
    if args.json:
        print(json.dumps({"query": args.query, "results": results}))
    else:
        for docid in results:
            print(docid)
    return 0


def _print_stats(root: Path, retriever_name: str, meta: Dict[str, Any]) -> None:
    stale = is_stale(root, meta)
    print(f"retriever: {retriever_name}")
    print(f"root: {meta.get('corpus_root')}")
    print(f"docs: {meta.get('doc_count')}")
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


_COMMANDS = {
    "index": _cmd_index,
    "query": _cmd_query,
    "stats": _cmd_stats,
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
