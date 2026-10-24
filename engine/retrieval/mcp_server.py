"""MCP (stdio) server exposing the engine's retrieval to an agent.

Started per project (``retrieval mcp`` / ``retrieval-mcp``), it offers three
tools -- ``search_domain_context``, ``index_project``, ``index_status`` -- so
an agent can ground a new implementation in the repo's existing code and
docs. No retrieval logic lives here: every handler delegates to
``retrieval.cli`` / ``retrieval.persistence`` and only adapts arguments,
staleness policy, snippets and JSON shape.

The optional ``mcp`` SDK is imported lazily in :func:`build_server`, so this
module (and its handlers) import and run with the stdlib-only core. stdout is
reserved for the MCP protocol: :func:`serve` repoints fd 1 at stderr and gives
the SDK a private copy of the original stdout.

Concurrency: ``index_status`` and the status resource are lock-free (they only
read small meta files; cache writes are atomic ``os.replace`` swaps).
``index_project`` holds ``_BUILD_LOCK`` for the whole build (a second build
returns a ``BuildInProgress`` error at once). Searches that may write
(lexical auto-refresh, ``build_missing``) serialize on ``_SEARCH_LOCK``; while a
build is running they never block or rebuild: they search the existing caches
read-only and say so (``build_in_progress``). A running build cannot be
cancelled from MCP; stop the server process to abort it.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import threading
from functools import partial
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from retrieval import __version__, cli, extractors
from retrieval.persistence import (
    cache_is_loadable,
    index_dir,
    is_stale,
    load_index,
    loader_kw_from_meta,
)
from retrieval.retrievers import Retriever

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

_PI_SERINI = "pi-serini"
# A concurrent build can replace pi-serini's Lucene dir under a loaded span map.
_PI_BUILD_SKIP = {"name": _PI_SERINI, "reason": "an index build is in progress and may be "
                  "replacing the pi-serini index; retry when it finishes"}
_BUILD_LOCK = threading.Lock()   # held for the duration of one index_project build
_SEARCH_LOCK = threading.Lock()  # serializes searches that may write caches
_RETRIEVER_CACHE: Dict[Tuple[str, str], Tuple[str, Retriever]] = {}
_RETRIEVER_CACHE_MUTEX = threading.Lock()
_CTX = "lexical+ctx"
_SHARED_SLOT = ("lexical", _CTX)  # one cache slot (see persistence._CACHE_LAYOUT)
_MCP_INDEX_CHOICES = tuple(c for c in cli._INDEX_RETRIEVER_CHOICES if c != _CTX)
_CTX_COST_HINT = (
    "lexical+ctx issues one LLM call per chunk (cost): ask the user for explicit opt-in, "
    "then run `retrieval index --retriever lexical+ctx` via the CLI. It is never "
    "built or refreshed through MCP."
)
_MAX_TOP_K = 25  # clamp, not reject: an over-eager agent still gets an answer
_MAX_SNIPPET_BYTES = 4000
STATUS_RESOURCE_URI = "retrieval://index/status"
_NOT_INDEXED = "not indexed - call index_project(retriever='all'|'{name}')"


def resolve_server_root(root_arg: Optional[str]) -> Path:
    """``--root`` > ``RETRIEVAL_ROOT`` > ``CLAUDE_PROJECT_DIR`` > cwd."""
    for candidate in (root_arg, os.environ.get("RETRIEVAL_ROOT"),
                      os.environ.get("CLAUDE_PROJECT_DIR")):
        if candidate:
            return Path(candidate).resolve()
    return Path.cwd().resolve()


# --------------------------------------------------------------------- helpers


def _weights_arg(weights: Optional[Dict[str, float]]) -> Optional[str]:
    """Validate and serialize *weights* to the CLI's ``"k:v,k:v"`` form."""
    if not weights:
        return None
    unknown = sorted(set(weights) - set(cli._RETRIEVER_CHOICES))
    if unknown:
        raise ValueError(
            f"unknown weight key(s) {unknown}; valid: {list(cli._RETRIEVER_CHOICES)}"
        )
    return ",".join(f"{name}:{float(value)}" for name, value in weights.items())


def _query_namespace(
    root: Path, query: str, retriever: str, top_k: Optional[int], pool: Optional[int],
    weights: Optional[str], stale_ok: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        root=str(root), query=query, retriever=retriever, top_k=top_k, auto=False,
        pool=pool, json=False, stale_ok=stale_ok, weights=weights, output=None,
    )


def _index_namespace(root: Path, retriever: str, force: bool, auto: bool) -> argparse.Namespace:
    overrides = {name: None for name in cli._INDEX_OVERRIDE_ARGS}
    return argparse.Namespace(
        root=str(root), retriever=retriever, force=force, auto=auto,
        no_pdf=False, allow_large_context=False, **overrides,
    )


def _slot_is_ctx(root: Path) -> bool:
    """True when the shared lexical cache slot was built as (paid) lexical+ctx."""
    meta = cli.read_meta(root, "lexical")
    return bool(meta and meta.get("retriever_name") == _CTX)


def _lexical_stale(root: Path, memo: Dict[str, str]) -> bool:
    """Stale *plain* lexical cache; a lexical+ctx slot is never auto-refreshed."""
    meta = cli.read_meta(root, "lexical")
    if meta is None or meta.get("retriever_name") == _CTX or not cache_is_loadable(root, "lexical"):
        return False
    return is_stale(root, meta, fingerprint_memo=memo, **loader_kw_from_meta(meta))


def _refresh_lexical_if_stale(root: Path, memo: Dict[str, str]) -> None:
    """Rebuild only the (cheap) plain-lexical cache when it is stale."""
    if _lexical_stale(root, memo):
        cli._load_or_rebuild(root, "lexical", False)


class _CachedLoader:
    """``load`` hook for ``cli._query_all/_query_single``: reuses deserialized
    retrievers across queries (keyed by the full meta, so any rebuild evicts),
    shares one fingerprint *memo*, never builds or rebuilds a lexical+ctx slot
    (even when its data file is corrupt), and in *read_only* mode (a build is
    running) never builds or rebuilds anything."""

    def __init__(self, root: Path, memo: Dict[str, str], read_only: bool) -> None:
        self.root = root
        self.memo = memo
        self.read_only = read_only

    def __call__(self, _root: Path, name: str, stale_ok: bool) -> Retriever:
        meta = cli.read_meta(self.root, name)
        if meta is None or not cache_is_loadable(self.root, name):
            return self._rebuild(name, stale_ok)
        if self._is_paid(name, meta):
            stale_ok = True
        if not (stale_ok or self.read_only) and is_stale(
            self.root, meta, fingerprint_memo=self.memo, **loader_kw_from_meta(meta)
        ):
            return self._rebuild(name, False)
        key = (str(index_dir(self.root)), name)
        stamp = json.dumps(meta, sort_keys=True, default=repr)
        with _RETRIEVER_CACHE_MUTEX:
            hit = _RETRIEVER_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        loaded = load_index(self.root, name)
        if loaded is None:
            return self._rebuild(name, stale_ok)
        with _RETRIEVER_CACHE_MUTEX:
            _RETRIEVER_CACHE[key] = (stamp, loaded[0])
        return loaded[0]

    @staticmethod
    def _is_paid(name: str, meta: Optional[Dict[str, Any]]) -> bool:
        return name in _SHARED_SLOT and bool(meta and meta.get("retriever_name") == _CTX)

    def _rebuild(self, name: str, stale_ok: bool) -> Retriever:
        """Fallback build; refused for read-only mode and for any paid slot."""
        if self.read_only:
            raise RuntimeError(f"{name}: not available while an index build is in progress")
        if name == _CTX or (
            name in _SHARED_SLOT and self._is_paid(name, cli._peek_meta(self.root, name))
        ):
            raise RuntimeError(
                f"{name}: the lexical+ctx index is unusable and is never rebuilt through "
                "MCP (LLM cost); rebuild it explicitly with the CLI"
            )
        return cli._load_or_rebuild(self.root, name, stale_ok)


def _cached_loader(
    root: Path, memo: Dict[str, str], read_only: bool
) -> Callable[[Path, str, bool], Retriever]:
    return _CachedLoader(root, memo, read_only)


def _available_names(root: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Strategies safe to consolidate without building: lexical (built lazily)
    plus any heavier strategy whose cache already exists and is loadable."""
    names: List[str] = []
    skipped: List[Dict[str, str]] = []
    for name in cli._DEFAULT_INDEX_SET:
        if name == "lexical" or cache_is_loadable(root, name):
            names.append(name)
        else:
            skipped.append({"name": name, "reason": _NOT_INDEXED.format(name=name)})
    return names, skipped


def _stale_names(root: Path, names: Sequence[str], memo: Dict[str, str]) -> List[str]:
    stale = []
    for name in names:
        meta = cli.read_meta(root, name)
        if meta is not None and cache_is_loadable(root, name) and is_stale(
            root, meta, fingerprint_memo=memo, **loader_kw_from_meta(meta)
        ):
            stale.append(name)
    return stale


class _SnippetFiles:
    """Per-request cache of each file's leading lines, so several hits in one
    file read it once and a hit never reads past its own window."""

    def __init__(self) -> None:
        self._lines: Dict[Path, Tuple[List[str], bool]] = {}

    def window(self, target: Path, first: int, last: int) -> List[str]:
        lines, complete = self._lines.get(target, ([], False))
        if len(lines) < last and not complete:
            with target.open(encoding="utf-8", errors="replace") as handle:
                lines = [line.rstrip("\n") for line in islice(handle, last)]
            complete = len(lines) < last
            self._lines[target] = (lines, complete)
        return lines[first - 1:last]


def _read_snippet(
    root: Path, rel_path: str, start: Optional[int], end: Optional[int], max_lines: int,
    files: Optional[_SnippetFiles] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(snippet, note)``; snippet is None when it cannot be shown."""
    if Path(rel_path).suffix.lower() in extractors.EXTRACTABLE_EXTENSIONS:
        return None, "media source; spans refer to its sidecar transcript"
    root_resolved = root.resolve()
    target = (root_resolved / rel_path).resolve()
    if not target.is_relative_to(root_resolved):
        return None, "path escapes project root"
    first = max(start or 1, 1)
    last = min(end if end is not None else first + max_lines - 1, first + max_lines - 1)
    try:
        window = (files or _SnippetFiles()).window(target, first, last)
    except OSError:
        return None, "file unreadable"
    return "\n".join(window), None


def _cap_bytes(text: str) -> Tuple[str, bool]:
    """Cap *text* to ``_MAX_SNIPPET_BYTES`` UTF-8 bytes; flag whether it was cut."""
    raw = text.encode("utf-8")
    if len(raw) <= _MAX_SNIPPET_BYTES:
        return text, False
    return raw[:_MAX_SNIPPET_BYTES].decode("utf-8", errors="ignore"), True


def _attach_snippets(
    root: Path, rows: List[Dict[str, Any]], include: bool, max_lines: int
) -> None:
    files = _SnippetFiles()
    for row in rows:
        if not include:
            row["snippet"] = None
            row["truncated"] = False
            continue
        snippet, note = _read_snippet(
            root, row["path"], row.get("start_line"), row.get("end_line"), max_lines, files
        )
        span_cut = (row.get("start_line") is not None and row.get("end_line") is not None
                    and row["end_line"] - row["start_line"] + 1 > max_lines)
        bytes_cut = False
        if snippet is not None:
            snippet, bytes_cut = _cap_bytes(snippet)
        row["snippet"] = snippet
        row["truncated"] = bool(snippet is not None and (bytes_cut or span_cut))
        if note:
            row["snippet_note"] = note


def _hint(
    results: List[Dict[str, Any]], skipped: List[Dict[str, str]], stale: List[str],
    used: Sequence[str] = (), ctx_stale: bool = False, building: bool = False,
) -> str:
    parts = []
    if not used:
        # Nothing was searched: absence of results says nothing about precedent.
        reasons = ("; ".join(f"{s['name']}: {s['reason']}" for s in skipped)
                   or "no strategy available")
        parts.append(
            f"The search did not run ({reasons}). This does NOT mean the repo has no "
            "precedent. Fix: index_project(retriever='lexical') or search with "
            "retriever='lexical', then retry; use Grep/Read meanwhile."
        )
    elif not results:
        parts.append("No matches; reformulate once with identifiers or domain terms the code "
                     "would use. If it still finds nothing, the repo likely has no precedent: "
                     "say so rather than guessing.")
    else:
        parts.append("Read the returned path:start_line-end_line spans before implementing.")
    if skipped and used:
        parts.append(
            f"Searched only: {', '.join(used)}; not searched: "
            f"{', '.join(s['name'] for s in skipped)} (see 'skipped'). Call "
            "index_project(retriever='all') for dense/AST/Lucene ranking (slow, one-off)."
        )
    plain_stale = [n for n in stale if not (ctx_stale and n in _SHARED_SLOT)]
    if plain_stale:
        parts.append(f"Stale caches ({', '.join(plain_stale)}) were searched as-is; "
                     "call index_project to refresh.")
    if ctx_stale:
        parts.append(f"The lexical+ctx index is stale and was searched as-is; {_CTX_COST_HINT}")
    if building:
        parts.append("An index build is in progress: existing caches were searched "
                     "read-only and nothing was rebuilt.")
    return " ".join(parts)


def _validate_search(query: str, top_k: Optional[int], pool: Optional[int],
                     snippet_lines: int) -> None:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if pool is not None and not 1 <= pool <= 500:
        raise ValueError("pool must be between 1 and 500")
    if not 1 <= snippet_lines <= 200:
        raise ValueError("snippet_lines must be between 1 and 200")


# -------------------------------------------------------------------- handlers


def _resolve_stale_ok(
    root: Path, stale_ok: Optional[bool], retriever: str, memo: Dict[str, str]
) -> bool:
    """None = default policy: refresh only the cheap plain-lexical cache (and only
    when lexical is being searched), keep heavy caches as-is. An explicit False
    rebuilds stale caches; True never rebuilds."""
    if stale_ok is None:
        if retriever in ("all", "lexical"):
            _refresh_lexical_if_stale(root, memo)
        return True
    return stale_ok


def _search_consolidated(
    root: Path, ns_args: Dict[str, Any], stale_ok: Optional[bool], build_missing: bool,
    memo: Dict[str, str], load: Callable[[Path, str, bool], Retriever], read_only: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, str]], List[str]]:
    if build_missing:
        names: Sequence[str] = cli._DEFAULT_INDEX_SET
        pre_skipped: List[Dict[str, str]] = []
    else:
        names, pre_skipped = _available_names(root)
    if read_only and _PI_SERINI in names:
        names = [n for n in names if n != _PI_SERINI]
        pre_skipped = pre_skipped + [_PI_BUILD_SKIP]
    effective = True if read_only else _resolve_stale_ok(root, stale_ok, "all", memo)
    out = cli._query_all(root, _query_namespace(root, stale_ok=effective, **ns_args), names, load)
    rows = [cli._consolidated_hit_to_json(h) for h in out["results"]]
    stale = _stale_names(root, out["retrievers"], memo)
    return out, rows, pre_skipped + out["skipped"], stale


def _search_single(
    root: Path, ns_args: Dict[str, Any], stale_ok: Optional[bool], build_missing: bool,
    memo: Dict[str, str], load: Callable[[Path, str, bool], Retriever], read_only: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[str]]:
    """Return ``(rows, skipped, stale)``; an unusable strategy yields no rows and
    an actionable ``skipped`` reason instead of raising."""
    name = ns_args["retriever"]
    if read_only and name == _PI_SERINI:
        return [], [_PI_BUILD_SKIP], []
    if name != "lexical" and not build_missing and not cache_is_loadable(root, name):
        reason = _NOT_INDEXED.format(name=name) + " (or use retriever='lexical')"
        return [], [{"name": name, "reason": reason}], []
    effective = True if read_only else _resolve_stale_ok(root, stale_ok, name, memo)
    ns = _query_namespace(root, stale_ok=effective, **ns_args)
    try:
        hits = cli._query_single(root, ns, load)
    except RuntimeError as exc:
        reason = f"{exc} - fall back to retriever='lexical'"
        return [], [{"name": name, "reason": reason}], []
    return [cli._hit_to_json(h) for h in hits], [], _stale_names(root, [name], memo)


def _index_block(root: Path, names: Sequence[str], stale: Sequence[str]) -> Dict[str, Any]:
    """Freshness of the caches a search used: ``stale`` and newest ``created_at``."""
    created = [m["created_at"] for m in (cli.read_meta(root, n) for n in names)
               if m and m.get("created_at")]
    return {"stale": bool(stale), "created_at": max(created) if created else None}


def _opt_in_error(error: str) -> Dict[str, Any]:
    return {"error": error, "error_type": "OptInRequired", "hint": _CTX_COST_HINT}


def _ctx_search_refusal(
    root: Path, stale_ok: Optional[bool], memo: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Searching lexical+ctx is allowed only against an existing index that needs no
    rebuild; anything else would spend LLM tokens without the CLI's opt-in guard."""
    meta = cli.read_meta(root, _CTX)
    if meta is None or meta.get("retriever_name") != _CTX or not cache_is_loadable(root, _CTX):
        return _opt_in_error("no lexical+ctx index exists for this project")
    if stale_ok is False and is_stale(root, meta, fingerprint_memo=memo,
                                      **loader_kw_from_meta(meta)):
        return _opt_in_error("the lexical+ctx index is stale and stale_ok=False would rebuild it")
    return None


def _run_search(
    root: Path, ns_args: Dict[str, Any], stale_ok: Optional[bool], build_missing: bool,
    read_only: bool, memo: Dict[str, str],
) -> Tuple[str, List[str], List[Dict[str, Any]], List[Dict[str, str]], List[str]]:
    load = _cached_loader(root, memo, read_only)
    build_missing = build_missing and not read_only
    retriever = ns_args["retriever"]
    if retriever == "all":
        out, rows, skipped, stale = _search_consolidated(
            root, ns_args, stale_ok, build_missing, memo, load, read_only)
        return "consolidated", out["retrievers"], rows, skipped, stale
    rows, skipped, stale = _search_single(
        root, ns_args, stale_ok, build_missing, memo, load, read_only)
    return "single", ([] if skipped else [retriever]), rows, skipped, stale


def search_domain_context(
    root: Path,
    query: str,
    retriever: str = "all",
    top_k: Optional[int] = None,
    pool: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    stale_ok: Optional[bool] = None,
    build_missing: bool = False,
    include_snippet: bool = True,
    snippet_lines: int = 20,
) -> Dict[str, Any]:
    """Find this repo's existing code/docs relevant to *query*. Never returns an empty
    envelope silently: unusable strategies are listed in ``skipped`` with a fix."""
    _validate_search(query, top_k, pool, snippet_lines)
    if retriever not in cli._QUERY_RETRIEVER_CHOICES:
        raise ValueError(f"unknown retriever {retriever!r}; valid: "
                         f"{list(cli._QUERY_RETRIEVER_CHOICES)}")
    memo: Dict[str, str] = {}  # one fingerprint walk per request
    if retriever == _CTX:
        refusal = _ctx_search_refusal(root, stale_ok, memo)
        if refusal is not None:
            return refusal
    top_k = min(top_k, _MAX_TOP_K) if top_k is not None else None
    ns_args = dict(query=query, retriever=retriever, top_k=top_k, pool=pool,
                   weights=_weights_arg(weights))
    building = _BUILD_LOCK.locked()
    if not building:
        with _SEARCH_LOCK:
            # Re-check under the lock: index_project takes _BUILD_LOCK then uses
            # _SEARCH_LOCK as a barrier, so no cache write overlaps a build.
            building = _BUILD_LOCK.locked()
            if not building:
                mode, used, rows, skipped, stale = _run_search(
                    root, ns_args, stale_ok, build_missing, False, memo)
    if building:
        mode, used, rows, skipped, stale = _run_search(
            root, ns_args, stale_ok, build_missing, True, memo)
    _attach_snippets(root, rows, include_snippet, snippet_lines)
    ctx_stale = bool(set(stale) & set(_SHARED_SLOT)) and _slot_is_ctx(root)
    return {
        "root": str(root), "query": query, "mode": mode, "retriever": retriever,
        "retrievers": used, "skipped": skipped, "stale": stale,
        "build_in_progress": building,
        "index": _index_block(root, used, stale), "results": rows,
        "hint": _hint(rows, skipped, stale, used, ctx_stale, building),
    }


def index_status(root: Path) -> Dict[str, Any]:
    """Report which strategy caches exist, load, and are stale."""
    memo: Dict[str, str] = {}
    strategies = []
    for name in cli._DEFAULT_INDEX_SET:
        meta = cli._peek_meta(root, name)
        loadable = cache_is_loadable(root, name)
        stale: Optional[bool] = None
        if meta is not None and loadable:
            stale = is_stale(root, meta, fingerprint_memo=memo, **loader_kw_from_meta(meta))
        strategies.append({
            "name": name, "indexed": meta is not None, "loadable": loadable, "stale": stale,
            "built_as": (meta or {}).get("retriever_name"),
            "chunks": (meta or {}).get("doc_count"), "files": (meta or {}).get("file_count"),
            "created_at": (meta or {}).get("created_at"),
            "engine_version": (meta or {}).get("engine_version"),
        })
    ready = [s["name"] for s in strategies if s["loadable"] and s["stale"] is False]
    return {
        "root": str(root), "cache_dir": str(index_dir(root)), "engine_version": __version__,
        "strategies": strategies, "ready": ready,
        "recommendation": _recommendation(strategies),
    }


def _recommendation(strategies: List[Dict[str, Any]]) -> str:
    by_name = {s["name"]: s for s in strategies}
    if not by_name["lexical"]["indexed"]:
        return ("Nothing indexed yet: call index_project(retriever='lexical') (fast), or just "
                "search_domain_context (builds lexical lazily). index_project(retriever='all') "
                "adds dense/AST/Lucene strategies (slow on large repos).")
    stale = [s["name"] for s in strategies if s["stale"]]
    ctx_stale = "lexical" in stale and by_name["lexical"]["built_as"] == _CTX
    stale = [n for n in stale if not (ctx_stale and n == "lexical")]
    if ctx_stale:
        return ("Stale lexical+ctx index (searched as-is, never auto-refreshed): "
                + _CTX_COST_HINT + (f" Also stale: {', '.join(stale)}." if stale else ""))
    if stale:
        return f"Stale: {', '.join(stale)}. Call index_project to refresh."
    missing = [s["name"] for s in strategies if not s["loadable"]]
    if missing:
        return (f"Not available: {', '.join(missing)}. index_project(retriever='all') builds "
                "those whose extras are installed.")
    return "All strategies are indexed and fresh."


def index_project(
    root: Path, retriever: str = "all", force: bool = False, auto: bool = False
) -> Dict[str, Any]:
    """Build/refresh caches via the engine's ``index`` command (never lexical+ctx)."""
    if retriever == _CTX:
        return _opt_in_error("lexical+ctx cannot be built through MCP")
    if retriever not in _MCP_INDEX_CHOICES:
        raise ValueError(f"unknown retriever {retriever!r}; valid: {list(_MCP_INDEX_CHOICES)}")
    if retriever in ("lexical", "all") and _slot_is_ctx(root):
        # lexical shares its cache slot with lexical+ctx: building it would wipe the paid index.
        return _opt_in_error(
            "this project's lexical slot holds a lexical+ctx index; building "
            f"{retriever!r} would overwrite it (for 'all', index the heavier strategies "
            "one by one instead)")
    if not _BUILD_LOCK.acquire(blocking=False):
        return {"error": "an index build is already running", "error_type": "BuildInProgress",
                "hint": "Call index_status (never blocked) and retry when it finishes; "
                        "searches meanwhile use the existing caches."}
    try:
        with _SEARCH_LOCK:
            pass  # barrier: wait out any in-flight cache-writing search
        with _diverted(), _captured() as buf:
            exit_code = cli._cmd_index(_index_namespace(root, retriever, force, auto))
    finally:
        _BUILD_LOCK.release()
    return {
        "root": str(root), "retriever": retriever, "force": force, "exit_code": exit_code,
        "ok": exit_code == 0,
        "log": [line for line in buf.getvalue().splitlines() if line.strip()],
        "status": index_status(root),
    }


# ---------------------------------------------------------------------- server


class _Router(io.TextIOBase):
    """Stand-in for sys.stdout/sys.stderr: writes go to the calling thread's
    capture buffer if one is set, else to the real stderr (never the protocol)."""

    def __init__(self, default: Any) -> None:
        self._default = default

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        sink = getattr(_sink, "buf", None)
        return (sink if sink is not None else self._default).write(text)

    def flush(self) -> None:
        self._default.flush()


_sink = threading.local()
_divert_mutex = threading.Lock()
_divert_state: Dict[str, Any] = {"depth": 0, "saved": None}


@contextlib.contextmanager
def _diverted() -> Iterator[None]:
    """Reference-counted, thread-safe replacement of sys.stdout/sys.stderr (plain
    ``redirect_stdout`` would corrupt sys.stdout when handlers overlap)."""
    with _divert_mutex:
        if _divert_state["depth"] == 0:
            _divert_state["saved"] = (sys.stdout, sys.stderr)
            sys.stdout = sys.stderr = _Router(sys.stderr)
        _divert_state["depth"] += 1
    try:
        yield
    finally:
        with _divert_mutex:
            _divert_state["depth"] -= 1
            if _divert_state["depth"] == 0:
                sys.stdout, sys.stderr = _divert_state["saved"]


@contextlib.contextmanager
def _captured() -> Iterator[io.StringIO]:
    """Capture this thread's stdout+stderr (needs an active ``_diverted``)."""
    buf, previous = io.StringIO(), getattr(_sink, "buf", None)
    _sink.buf = buf
    try:
        yield buf
    finally:
        _sink.buf = previous


def _guarded(fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run an engine handler with stray prints kept off the protocol stdout;
    exceptions become ``{"error": ...}`` dicts so the agent can act on them."""
    try:
        with _diverted():
            return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - tool boundary: report, never crash the server
        return {
            "error": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
            "hint": "Fix the arguments or call index_status; fall back to the "
                    "`retrieval query` CLI or Grep/Read if the problem persists.",
        }


_INSTRUCTIONS = (
    "Domain-context grounding for THIS project. USE search_domain_context FIRST before "
    "implementing, changing or debugging code here, and before answering from memory: it "
    "returns evidence spans (path:start_line-end_line + snippet) from this repo, not an "
    "answer. Read the cited paths and follow the existing patterns; cite spans you relied "
    "on; if nothing relevant exists, say the repo has no precedent. Call index_status at "
    "session start. Only lexical (keyword) search is available until index_project("
    "retriever='all') builds the dense, AST and Lucene strategies (slow on large repos; "
    "one-off). Prompt ground_implementation(task) runs this workflow."
)

_SEARCH_DESC = (
    "USE THIS FIRST before implementing, changing or debugging code in this project, and "
    "before answering from memory: returns evidence spans (path:start_line-end_line + "
    "snippet) from this repo, not an answer. Read the cited paths and follow the existing "
    "patterns. If results are weak, reformulate once with identifiers/domain terms; if "
    "still empty, the repo has no precedent - say so.\n\n"
    "retriever: 'all' (default, consolidated) or lexical|lexical+ctx|turbovec|pi-serini|"
    "hybrid|treesitter (lexical+ctx only searches an existing index; never built here). "
    "top_k 1..25 (default 5; larger is clamped). pool 1..500 "
    "(consolidated only, deeper per-retriever candidates for recall). weights: "
    "{retriever: weight}. stale_ok: None=auto (refresh only lexical), True=never rebuild, "
    "False=rebuild stale caches. build_missing=True builds every missing strategy (slow). "
    "Never raises: see 'skipped' (with fixes), 'index' {stale, created_at} and 'error'."
)

_WORKFLOW_PROMPT = """\
Ground this task in the repository's existing code before writing anything.

Task: {task}

1. Call search_domain_context with a natural-language description of the task plus key \
identifiers. If 'skipped' lists unindexed strategies and recall matters, consider \
index_project(retriever='all').
2. Read the cited path:start_line-end_line spans (use the Read tool) - snippets are leads, \
not the full context.
3. If results are weak, reformulate once or twice (different identifiers, synonyms, \
neighbouring concepts, or retriever='treesitter' for code / 'turbovec' for paraphrase).
4. Implement following the existing patterns, naming and error handling you found.
5. Cite the spans (path:start-end) you relied on. If the repo has no precedent, say so \
explicitly instead of inventing one."""


def ground_implementation(task: str) -> str:
    """Workflow prompt text for grounding *task* in this repo."""
    return _WORKFLOW_PROMPT.format(task=task.strip() or "(describe the task)")


def build_server(root: Path) -> "FastMCP":
    """Create the FastMCP server bound to *root* (imports ``mcp`` lazily)."""
    import anyio.to_thread
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    writes_cache = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                   idempotentHint=True, openWorldHint=False)

    server = FastMCP("agentic-retrieval", instructions=_INSTRUCTIONS)

    @server.tool(name="search_domain_context", description=_SEARCH_DESC,
                 annotations=writes_cache)
    async def _tool_search(
        query: str,
        retriever: str = "all",
        top_k: Optional[int] = None,
        pool: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        stale_ok: Optional[bool] = None,
        build_missing: bool = False,
        include_snippet: bool = True,
        snippet_lines: int = 20,
    ) -> Dict[str, Any]:
        return await anyio.to_thread.run_sync(partial(
            _guarded, search_domain_context, root, query, retriever, top_k, pool,
            weights, stale_ok, build_missing, include_snippet, snippet_lines,
        ))

    @server.tool(name="index_project", annotations=writes_cache)
    async def _tool_index(
        retriever: str = "all", force: bool = False, auto: bool = False
    ) -> Dict[str, Any]:
        """Build or refresh retrieval caches (heavier strategies: turbovec, hybrid,
        treesitter, pi-serini; 'all' is slow on big repos but one-off). Never builds
        lexical+ctx (LLM cost; CLI opt-in only). One build at a time. Never raises."""
        return await anyio.to_thread.run_sync(partial(
            _guarded, index_project, root, retriever, force, auto))

    @server.tool(name="index_status", annotations=read_only)
    async def _tool_status() -> Dict[str, Any]:
        """Report which strategy caches exist, are loadable and stale, with a
        recommendation. Call at session start. Never raises."""
        return await anyio.to_thread.run_sync(partial(
            _guarded, index_status, root))

    @server.resource(STATUS_RESOURCE_URI, name="index_status", mime_type="application/json",
                     description="Per-strategy index status (same payload as index_status).")
    async def _resource_status() -> str:
        return json.dumps(await anyio.to_thread.run_sync(partial(
            _guarded, index_status, root)))

    @server.prompt(name="ground_implementation",
                   description="Workflow: search this repo, read spans, follow existing "
                               "patterns, cite path:start-end.")
    def _prompt_ground(task: str) -> str:
        return ground_implementation(task)

    return server


def _protect_stdout() -> None:
    """Reserve the real stdout for the protocol: fd 1 (native/JVM writers) is
    repointed at stderr and ``sys.stdout`` becomes a private copy of the
    original fd 1 that the SDK's stdio transport wraps."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = io.TextIOWrapper(os.fdopen(proto_fd, "wb"), encoding="utf-8", line_buffering=True)


def serve(root: Path) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _protect_stdout()
    build_server(root).run(transport="stdio")


def run_cli(root_arg: Optional[str]) -> int:
    root = resolve_server_root(root_arg)
    try:
        import mcp  # noqa: F401
    except ImportError:
        print(
            "agentic-retrieval MCP server needs the 'mcp' extra: "
            "uv sync --project <engine> --extra mcp "
            "(or run via the plugin's scripts/retrieval-mcp.sh)",
            file=sys.stderr,
        )
        return 1
    print(f"agentic-retrieval MCP: root={root}", file=sys.stderr)
    serve(root)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-mcp")
    parser.add_argument(
        "--root", help="project root (default: RETRIEVAL_ROOT, CLAUDE_PROJECT_DIR, or cwd)"
    )
    return run_cli(parser.parse_args(argv).root)


if __name__ == "__main__":
    sys.exit(main())
