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
RuntimeError when its extras are missing). ``extract`` pre-warms PDF
sidecar transcripts via ``pypdf`` (the only subcommand that hard-fails on
a missing ``pdf`` extra); ``sidecar`` inspects sidecar state (``--list``)
or registers an agent-authored transcript (``--register``) without ever
needing ``pypdf`` on either mode — the no-install recovery path.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval import __version__, extractors
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
    loader_kw_from_meta,
    save_index,
)
from retrieval.project_loader import (
    DEFAULT_EXTENSIONS,
    discover_files,
    load_ast_chunk_documents,
    load_chunk_documents,
)
from retrieval.retrievers import PiSeriniRetriever, Retriever, build_retriever, resolve_ctor_kwargs

_RETRIEVER_CHOICES = (
    "lexical", "lexical+ctx", "turbovec", "pi-serini", "hybrid", "treesitter",
)
# lexical+ctx deliberately excluded: shares the lexical cache slot and costs LLM tokens.
_DEFAULT_INDEX_SET = ("lexical", "turbovec", "pi-serini", "hybrid", "treesitter")
_INDEX_RETRIEVER_CHOICES = _RETRIEVER_CHOICES + ("all",)
_QUERY_RETRIEVER_CHOICES = _RETRIEVER_CHOICES + ("all",)

#: lexical+ctx issues one LLM call per chunk-Document at index time, so a
#: large corpus can be slow/expensive; ``index --retriever lexical+ctx``
#: warns above ``_LEXICAL_CTX_WARN_CHUNKS`` and refuses (absent
#: ``--allow-large-context``) above ``_LEXICAL_CTX_HARD_LIMIT_CHUNKS``.
_LEXICAL_CTX_WARN_CHUNKS = 500
_LEXICAL_CTX_HARD_LIMIT_CHUNKS = 2000


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
    index_parser.add_argument(
        "--allow-large-context", action="store_true",
        help="skip the 'lexical+ctx' large-corpus guard (see "
        f"_LEXICAL_CTX_HARD_LIMIT_CHUNKS={_LEXICAL_CTX_HARD_LIMIT_CHUNKS}); "
        "each chunk costs one LLM call at index time",
    )
    index_parser.add_argument(
        "--no-pdf", action="store_true",
        help="exclude PDFs and all other sidecar-routed media (docx/pptx/"
        "xlsx/images) from discovery (escape hatch for the default "
        "auto-activated sidecar-extraction pipeline); sticky across "
        "later flag-less 'index'/'query' calls via the persisted meta",
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

    extract_parser = subparsers.add_parser(
        "extract",
        help="pre-warm PDF sidecar transcripts for a project root, without "
        "building/touching any retriever index",
    )
    extract_parser.add_argument(
        "--root", help="project root to extract from (default: RETRIEVAL_ROOT or cwd)"
    )
    extract_parser.add_argument(
        "--force", action="store_true",
        help="re-extract every PDF even if its cached sidecar is already fresh",
    )
    extract_parser.add_argument(
        "--prune", action="store_true",
        help="remove manifest entries and sidecar files for PDFs no longer "
        "present under --root",
    )
    extract_parser.add_argument(
        "--json", action="store_true", help="emit a JSON summary instead of one line per file"
    )

    sidecar_parser = subparsers.add_parser(
        "sidecar",
        help="inspect or author PDF/media sidecar transcripts without "
        "pypdf (no-install alternative to 'extract' for PDFs; the only "
        "indexing path for docx/pptx/xlsx/images; never imports pypdf)",
    )
    sidecar_parser.add_argument(
        "--root", help="project root (default: RETRIEVAL_ROOT or cwd)"
    )
    sidecar_parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of text"
    )
    sidecar_mode = sidecar_parser.add_mutually_exclusive_group(required=True)
    sidecar_mode.add_argument(
        "--list", action="store_true",
        help="report every discovered PDF/media file's sidecar state "
        "(missing/outdated/agent-authored/stub/ok)",
    )
    sidecar_mode.add_argument(
        "--register", metavar="SOURCE",
        help="register an agent-authored transcript as SOURCE's sidecar "
        "(requires --transcript)",
    )
    sidecar_parser.add_argument(
        "--transcript",
        help="path to the transcript file to register, or '-' for stdin "
        "(required with --register)",
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


def _loader_kw_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve the ``discover_files``/``compute_fingerprint`` loader kwargs
    an ``index``-time invocation of *args* implies.

    ``{}`` unless ``--no-pdf`` was given (``args`` lacks ``no_pdf`` entirely
    on subparsers other than ``index``, e.g. ``eval``), in which case PDFs
    *and every other sidecar-routed media suffix* (docx/pptx/xlsx/images)
    are excluded from discovery for this run. Persisted verbatim into meta
    via ``save_index``'s ``loader_kw`` and rehydrated by
    ``persistence.loader_kw_from_meta`` on every later flag-less
    ``index``/``query`` call, so passing ``--no-pdf`` once at index time is
    sticky.
    """
    if getattr(args, "no_pdf", False):
        return {"extensions": frozenset(DEFAULT_EXTENSIONS - extractors.EXTRACTABLE_EXTENSIONS)}
    return {}


def _documents_for(
    root: Path,
    retriever_name: str,
    policy: Optional[ChunkingPolicy] = None,
    loader_kw: Optional[Dict[str, Any]] = None,
):
    """Return the chunk-granularity Documents to index for *retriever_name*.

    ``treesitter`` uses AST-boundary chunks (carrying a ``context``
    breadcrumb); every other retriever uses the line-based chunker. *policy*
    (default ``None`` -> ``DEFAULT_POLICY``, today's flat 400-char chunking)
    picks each file's chunk-size bucket by suffix. *loader_kw* is forwarded
    to ``discover_files`` (via the loader functions' ``**kw``).
    """
    if retriever_name == "treesitter":
        return load_ast_chunk_documents(root, policy=policy, **(loader_kw or {}))
    return load_chunk_documents(root, policy=policy, **(loader_kw or {}))


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
    root: Path, args: argparse.Namespace, loader_kw: Optional[Dict[str, Any]] = None
) -> "tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[ChunkingPolicy]]":
    """Resolve ``(params, corpus_stats, policy)`` for ``index``/``eval``
    with precedence explicit flag > auto-derived > static default.

    Returns ``(None, None, None)`` — reproducing today's behavior exactly,
    with no ``hyperparams``/``corpus_stats`` meta block written — when
    neither ``--auto`` nor any hyperparameter flag was given. *loader_kw*
    (default ``None`` -> ``{}``) is forwarded into ``collect_signals`` when
    ``--auto`` was given.
    """
    overrides = _overrides_from_args(args)
    if not args.auto and not any(v is not None for v in overrides.values()):
        return None, None, None

    corpus_stats: Optional[Dict[str, Any]] = None
    if args.auto:
        signals = collect_signals(root, loader_kw=loader_kw)
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
    loader_kw: Optional[Dict[str, Any]] = None,
) -> Retriever:
    """Build a fresh retriever over *root* and persist it; return the retriever."""
    fingerprint = compute_fingerprint(root, **(loader_kw or {}))
    retriever = _make_retriever(root, retriever_name, params)
    retriever.index(_documents_for(root, retriever_name, policy, loader_kw))
    save_index(
        retriever, root, fingerprint, retriever_name, __version__,
        params=params, corpus_stats=corpus_stats, loader_kw=loader_kw or None,
    )
    return retriever


def _up_to_date_message(
    root: Path,
    retriever_name: str,
    params: Optional[Dict[str, Any]] = None,
    loader_kw: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return a "fast path" message if a fresh, non-stale cache exists for
    *root* + *retriever_name*; ``None`` if there's no cache or it's stale
    (a rebuild is needed).
    """
    cached = load_index(root, retriever_name)
    if cached is None:
        return None
    _retriever, meta = cached
    if is_stale(root, meta, params=params, **(loader_kw or {})):
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
    loader_kw: Optional[Dict[str, Any]] = None,
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
        if not force and _up_to_date_message(root, name, params, loader_kw) is not None:
            print(f"{name}: up to date (use --force to rebuild)")
            continue
        build_params, build_corpus_stats, build_policy = params, corpus_stats, policy
        if build_params is None:
            build_params, build_corpus_stats, build_policy = _params_for_rebuild(root, name, None)
        try:
            retriever = _build_and_save(
                root, name, build_params, build_corpus_stats, build_policy, loader_kw
            )
            chunk_count = len(retriever.to_dict()["docids"])
            print(f"{name}: indexed {chunk_count} chunks")
        except RuntimeError as exc:
            print(f"{name}: skipped ({str(exc).splitlines()[0]})")
            if name == "lexical":
                lexical_failed = True
    print(f"-> {index_dir(root)}")
    return 1 if lexical_failed else 0


def _check_lexical_ctx_guard(
    retriever_name: str, documents: List[Any], allow_large_context: bool
) -> None:
    """Guard against an accidental large/expensive ``lexical+ctx`` index run
    (one LLM call per chunk-Document): warn to stderr above
    ``_LEXICAL_CTX_WARN_CHUNKS``, and refuse outright (unless
    *allow_large_context*) above ``_LEXICAL_CTX_HARD_LIMIT_CHUNKS``. A no-op
    for every other *retriever_name*.
    """
    if retriever_name != "lexical+ctx":
        return
    count = len(documents)
    if count > _LEXICAL_CTX_HARD_LIMIT_CHUNKS and not allow_large_context:
        raise RuntimeError(
            f"lexical+ctx would index {count} chunks (> "
            f"{_LEXICAL_CTX_HARD_LIMIT_CHUNKS}), issuing one LLM call per "
            "chunk; pass --allow-large-context to proceed anyway, or index "
            "a smaller root/subtree."
        )
    if count > _LEXICAL_CTX_WARN_CHUNKS:
        print(
            f"warning: lexical+ctx is about to index {count} chunks "
            "(one LLM call per chunk) — this may be slow/expensive",
            file=sys.stderr,
        )


def _cmd_index(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    loader_kw = _loader_kw_from_args(args)
    params, corpus_stats, policy = _resolve_index_params(root, args, loader_kw)
    if args.retriever == "all":
        return _index_all(root, args.force, params, corpus_stats, policy, loader_kw)
    if not args.force:
        message = _up_to_date_message(root, args.retriever, params, loader_kw)
        if message is not None:
            print(message)
            return 0
    # Nothing explicit was given (no --auto, no hyperparameter flag): a
    # routine refresh (with or without --force) must not silently revert
    # this retriever's own previously recorded hyperparameters to static
    # defaults — recover them from its existing meta, if any.
    if params is None:
        params, corpus_stats, policy = _params_for_rebuild(root, args.retriever, None)
    fingerprint = compute_fingerprint(root, **loader_kw)
    retriever = _make_retriever(root, args.retriever, params)
    documents = _documents_for(root, args.retriever, policy, loader_kw)
    _check_lexical_ctx_guard(args.retriever, documents, args.allow_large_context)
    retriever.index(documents)
    saved_dir = save_index(
        retriever, root, fingerprint, args.retriever, __version__,
        params=params, corpus_stats=corpus_stats, loader_kw=loader_kw or None,
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


def _rebuild_loader_kw(
    root: Path, retriever_name: str, meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loader kwargs for a query-triggered rebuild, derived from *meta* (or
    a best-effort ``_peek_meta`` lookup when *meta* wasn't already loaded) —
    never from CLI args, so ``query`` never needs a discovery-affecting flag
    of its own."""
    if meta is None:
        meta = _peek_meta(root, retriever_name) or {}
    return loader_kw_from_meta(meta)


def _load_or_rebuild(
    root: Path, retriever_name: str, stale_ok: bool, params: Optional[Dict[str, Any]] = None
) -> Retriever:
    """Load the cached retriever for *root*, rebuilding it if missing/stale.

    A rebuild (whether triggered by no cache at all or by a stale
    fingerprint/hyperparams) preserves the index's own previously recorded
    hyperparameters — see ``_params_for_rebuild`` — rather than silently
    reverting to static defaults just because the caller didn't pass
    explicit *params*. Loader kwargs (extensions/exclude_dirs/etc.) are
    likewise recovered from the index's own meta (see ``_rebuild_loader_kw``)
    rather than requiring a query-time flag.
    """
    cached = load_index(root, retriever_name)
    if cached is None:
        rebuild_params, rebuild_corpus_stats, rebuild_policy = _params_for_rebuild(
            root, retriever_name, params
        )
        rebuild_loader_kw = _rebuild_loader_kw(root, retriever_name)
        return _build_and_save(
            root, retriever_name, rebuild_params, rebuild_corpus_stats, rebuild_policy,
            rebuild_loader_kw,
        )
    retriever, meta = cached
    rebuild_loader_kw = loader_kw_from_meta(meta)
    if is_stale(root, meta, params=params, **rebuild_loader_kw) and not stale_ok:
        rebuild_params, rebuild_corpus_stats, rebuild_policy = _params_for_rebuild(
            root, retriever_name, params, meta
        )
        return _build_and_save(
            root, retriever_name, rebuild_params, rebuild_corpus_stats, rebuild_policy,
            rebuild_loader_kw,
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


def _discover_media(root: Path) -> List[Path]:
    """Every sidecar-routed file (PDF + ``extractors.AGENT_ONLY_EXTENSIONS``
    media) under *root*, via the default ``discover_files`` eligibility
    rules. Used wherever "every sidecar-eligible file" is the question:
    ``sidecar --list``/``--register``'s discoverability check, and
    ``_cmd_extract``'s prune keep-set (pruning must not delete a registered
    agent-only sidecar just because ``extract`` itself never touches it)."""
    return [
        path for path in discover_files(root)
        if path.suffix.lower() in extractors.EXTRACTABLE_EXTENSIONS
    ]


def _discover_pdfs(root: Path) -> List[Path]:
    """PDFs under *root* with a registered machine extractor, via the
    default ``discover_files`` eligibility rules. Used by ``_cmd_extract``'s
    extraction loop: ``extract`` stays machine/pypdf-only and must never
    attempt agent-only media (docx/pptx/xlsx/images), which have no
    extractor to run."""
    return [
        path for path in discover_files(root)
        if path.suffix.lower() in extractors.MACHINE_EXTRACTABLE_EXTENSIONS
    ]


def _extract_file_report(rel: str, sidecar: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
    """One ``retrieval extract`` progress record for *rel*'s *sidecar*
    result; *entry* is that source's manifest entry (for ``pages``, which
    ``extractors.Sidecar`` itself doesn't carry)."""
    return {
        "source": rel,
        "sidecar": sidecar.docid,
        "status": sidecar.status,
        "reason": sidecar.reason,
        "pages": entry.get("pages", 0),
        "chars": len(sidecar.text),
    }


def _prune_orphan_sidecars(root: Path, keep_rel: "set[str]") -> int:
    """Remove manifest entries + sidecar files whose source PDF is no longer
    in *keep_rel*; return the number pruned."""
    manifest = extractors.load_manifest(root)
    entries = manifest.get("entries", {})
    orphans = [rel for rel in entries if rel not in keep_rel]
    for rel in orphans:
        sidecar_rel = entries[rel].get("sidecar") or extractors.sidecar_relpath(rel)
        try:
            (root / sidecar_rel).unlink()
        except OSError:
            pass
    extractors.prune_manifest(root, keep_rel)
    return len(orphans)


def _print_extract_report(
    file_reports: List[Dict[str, Any]], pruned: int, as_json: bool, root: Path
) -> None:
    if as_json:
        print(json.dumps({"root": str(root), "files": file_reports, "pruned": pruned}))
        return
    for report in file_reports:
        if report["status"] == "ok":
            print(
                f"{report['source']} -> {report['sidecar']} "
                f"({report['pages']} pages, {report['chars']} chars)"
            )
        else:
            print(f"{report['source']}: stub ({report['reason']})")
    if pruned:
        print(f"pruned {pruned} orphaned sidecar(s)")


def _warn_overwriting_agent_sidecars(root: Path) -> None:
    """``extract --force`` warning: agent-authored sidecars are never
    auto-superseded by a later pypdf install (see the module docstring's
    supersede policy) — ``--force`` is the one deliberate escape hatch, so
    it warns before silently discarding hand-authored transcripts."""
    entries = extractors.load_manifest(root).get("entries", {})
    n_agent = sum(
        1 for entry in entries.values() if entry.get("authored_by") == extractors.AUTHORED_BY_AGENT
    )
    if n_agent:
        print(
            f"warning: overwriting {n_agent} agent-authored sidecar(s) with pypdf output",
            file=sys.stderr,
        )


def _cmd_extract(args: argparse.Namespace) -> int:
    """Pre-warm every PDF's sidecar transcript under *args.root*; never
    builds/touches a retriever index. Hard-fails (guidance ``RuntimeError``,
    caught by ``main``'s error boundary) only here, when the ``pdf`` extra
    isn't installed — ``index``/``query`` never do.
    """
    root = _resolve_root(args.root)
    extractors.require_extractors(extractors.EXTRACTABLE_EXTENSIONS)
    if args.force:
        _warn_overwriting_agent_sidecars(root)
    pdf_paths = _discover_pdfs(root)
    sidecars = {
        pdf_path.relative_to(root).as_posix(): extractors.ensure_sidecar(
            root, pdf_path, force=args.force
        )
        for pdf_path in pdf_paths
    }
    manifest_entries = extractors.load_manifest(root).get("entries", {})
    file_reports = [
        _extract_file_report(rel, sidecar, manifest_entries.get(rel, {}))
        for rel, sidecar in sidecars.items()
    ]
    # Keep-set is every discovered sidecar-eligible file, not just the PDFs
    # this loop extracted: agent-only media (docx/pptx/xlsx/images) never
    # goes through ensure_sidecar here, but a registered agent sidecar for
    # one must survive --prune just like a PDF's would.
    keep_rel = {path.relative_to(root).as_posix() for path in _discover_media(root)}
    pruned = _prune_orphan_sidecars(root, keep_rel) if args.prune else 0
    _print_extract_report(file_reports, pruned, args.json, root)
    return 0


def _sidecar_state_line(state: Dict[str, Any]) -> str:
    """One ``retrieval sidecar --list`` text-mode line for *state* (an entry
    from ``extractors.sidecar_states``)."""
    label = state["state"]
    if label == "missing":
        return f"{state['rel']}: missing -> needs transcript"
    if label == "outdated":
        return f"{state['rel']}: outdated -> source changed, needs re-extraction"
    if label == "stub":
        return f"{state['rel']}: stub ({state['reason']}) -> needs transcript"
    if label == "agent-authored":
        return f"{state['rel']}: agent-authored ({state['pages']} pages)"
    return f"{state['rel']}: ok ({state['pages']} pages)"


def _cmd_sidecar_list(args: argparse.Namespace, root: Path) -> int:
    """``retrieval sidecar --list``: report every discovered PDF/media
    file's sidecar state, without importing/needing pypdf."""
    media_paths = _discover_media(root)
    states = extractors.sidecar_states(root, media_paths)
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "backend_available": extractors.backend_available(),
                    "files": states,
                }
            )
        )
        return 0
    for state in states:
        print(_sidecar_state_line(state))
    return 0


def _read_transcript(transcript_arg: str) -> str:
    """Read a transcript from a file path, or stdin when *transcript_arg*
    is ``-`` (the CLI-conventional stdin marker)."""
    if transcript_arg == "-":
        return sys.stdin.read()
    return Path(transcript_arg).read_text(encoding="utf-8")


def _cmd_sidecar_register(args: argparse.Namespace, root: Path) -> int:
    """``retrieval sidecar --register SOURCE --transcript PATH|-``: the
    no-pypdf recovery path — an agent hand-authors the transcript and
    registers it directly, no backend install required.

    ``args.transcript`` is guaranteed non-``None`` here: ``main`` enforces
    the --register/--transcript pairing (an argparse-style ``parser.error``,
    exit 2) before dispatching to this function.
    """
    transcript = _read_transcript(args.transcript)
    source_path = Path(args.register)
    if not source_path.is_absolute():
        source_path = root / source_path
    sidecar = extractors.register_sidecar(root, source_path, transcript)

    discoverable = source_path.resolve() in {p.resolve() for p in _discover_media(root)}
    if not discoverable:
        print(
            f"note: {source_path} is registered but not discoverable by the "
            "default project loader (excluded dir, extension, or size cap) "
            "- it will not be indexed",
            file=sys.stderr,
        )

    manifest_entries = extractors.load_manifest(root).get("entries", {})
    rel = source_path.resolve().relative_to(root.resolve()).as_posix()
    pages = manifest_entries.get(rel, {}).get("pages", 0)
    if pages == 0:
        print(
            "note: no '## Page N' headings found in the transcript - page "
            "breadcrumbs will be absent from search-hit context",
            file=sys.stderr,
        )

    if args.json:
        print(json.dumps({"source": rel, "sidecar": sidecar.docid, "pages": pages}))
    else:
        print(f"{rel} -> {sidecar.docid} ({pages} pages)")
        print("reindex to pick up this sidecar: `retrieval index` (or just query)")
    return 0


def _cmd_sidecar(args: argparse.Namespace) -> int:
    """``retrieval sidecar``: inspect (``--list``) or author
    (``--register``) PDF sidecar transcripts, never importing/requiring
    pypdf on either path (unlike ``extract``)."""
    root = _resolve_root(args.root)
    if args.list:
        return _cmd_sidecar_list(args, root)
    return _cmd_sidecar_register(args, root)


_COMMANDS = {
    "index": _cmd_index,
    "query": _cmd_query,
    "stats": _cmd_stats,
    "eval": _cmd_eval,
    "tune": _cmd_tune,
    "extract": _cmd_extract,
    "sidecar": _cmd_sidecar,
}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on a handled runtime error.

    Argument-parsing errors (missing/invalid flags) are handled by argparse
    itself and exit the process directly (``SystemExit``), before this
    function's error handling is reached.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "sidecar" and args.register and not args.transcript:
        parser.error("sidecar --register requires --transcript PATH|-")
    try:
        return _COMMANDS[args.command](args)
    except Exception as exc:  # noqa: BLE001 - CLI top-level error boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
