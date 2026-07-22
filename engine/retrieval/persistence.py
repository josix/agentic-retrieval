"""On-disk persistence for fitted retrievers, keyed by project path.

Stdlib-only (``json``, ``hashlib``, ``os``, ``pathlib``, ``time``) — each
retriever's cache is a pair of JSON files (data + meta, e.g. ``lexical.json``
+ ``meta.json`` for the lexical family, ``hybrid.json`` + ``hybrid.meta.json``
for hybrid) under a per-project directory derived from the resolved, absolute
project root path, so different projects (and different checkouts of the same
repo) never collide — and different retrievers for the same project coexist.

The ``pi-serini`` retriever additionally keeps its binary Lucene segments in
a ``lucene/`` subdirectory of the same per-project cache dir; its JSON file
only points at them.

Staleness is detected via a **fingerprint**: a SHA-256 hash over every
discovered file's relative path, size, and mtime (nanoseconds). Any file
added, removed, resized, or touched changes the fingerprint, so
``is_stale`` catches content drift without re-reading file contents.

Writes are atomic (write to a ``.tmp`` sibling, then ``os.replace``) so a
crash mid-write never leaves a half-written cache file for a future
``load_index`` to trip over.

Warning: if ``RETRIEVAL_INDEX_DIR`` is pointed *inside* the indexed project
root, the cache's ``lexical.json``/``meta.json`` get swept up as documents
on the next index/query run (a feedback loop) — only a directory literally
named ``.cache`` is excluded by default, so any other cache dirname is
fair game for re-indexing.

Since retrievers now index chunk-granularity Documents (see
``retrieval.project_loader.load_chunk_documents``), a meta file's
``doc_count`` is a **chunk** count (one chunk-Document per indexed span,
docid ``"{path}:{start}-{end}"``), while ``file_count`` is the number of
distinct source files those chunks came from (``len(set(source_path for
each unit))``) — the two will usually differ once a file yields more than
one chunk.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from retrieval.project_loader import discover_files
from retrieval.retrievers import (
    HybridRetriever,
    LexicalRetriever,
    PiSeriniRetriever,
    Retriever,
    TurbovecRetriever,
)

#: Default on-disk location for the index cache; overridable via
#: ``RETRIEVAL_INDEX_DIR`` (see ``cache_base_dir``).
DEFAULT_BASE = Path.home() / ".cache" / "agentic-retrieval" / "indexes"

_LEXICAL_FILENAME = "lexical.json"
_META_FILENAME = "meta.json"

#: retriever name -> (data filename, meta filename, deserializer class).
#: ``lexical`` keeps the original filenames for backward compatibility, and
#: ``lexical+ctx`` shares them: its persisted state is plain LexicalRetriever
#: data (the LLM context is baked into the indexed text), so the two are one
#: cache slot — ``meta.json``'s ``retriever_name`` records which one built it.
_CACHE_LAYOUT = {
    "lexical": (_LEXICAL_FILENAME, _META_FILENAME, LexicalRetriever),
    "lexical+ctx": (_LEXICAL_FILENAME, _META_FILENAME, LexicalRetriever),
    "turbovec": ("turbovec.json", "turbovec.meta.json", TurbovecRetriever),
    "pi-serini": ("pi-serini.json", "pi-serini.meta.json", PiSeriniRetriever),
    "hybrid": ("hybrid.json", "hybrid.meta.json", HybridRetriever),
}


def _layout(retriever_name: str) -> Tuple[str, str, Any]:
    try:
        return _CACHE_LAYOUT[retriever_name]
    except KeyError:
        raise ValueError(
            f"unknown retriever {retriever_name!r}; choose from {list(_CACHE_LAYOUT)}"
        ) from None


def cache_base_dir() -> Path:
    """Return the base directory all project index caches live under.

    Honors the ``RETRIEVAL_INDEX_DIR`` environment variable override; falls
    back to ``DEFAULT_BASE``.
    """
    override = os.environ.get("RETRIEVAL_INDEX_DIR")
    if override:
        return Path(override)
    return DEFAULT_BASE


def project_key(root: "os.PathLike[str] | str") -> str:
    """Return a short, stable, filesystem-safe key for *root*.

    Derived from the SHA-256 of the resolved, absolute POSIX path, truncated
    to 16 hex characters — short enough for a directory name, long enough
    that collisions are not a practical concern.
    """
    resolved = Path(root).resolve().as_posix()
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def index_dir(root: "os.PathLike[str] | str") -> Path:
    """Return the per-project cache directory for *root* (may not exist yet)."""
    return cache_base_dir() / project_key(root)


def compute_fingerprint(root: "os.PathLike[str] | str", **loader_kw: Any) -> str:
    """Return a SHA-256 fingerprint of every discoverable file under *root*.

    One ``"relpath|st_size|st_mtime_ns\\n"`` line per file, in the same
    deterministic sorted order ``discover_files`` returns, hashed together —
    so any file addition, removal, resize, or mtime change (edit) alters the
    fingerprint, without reading file contents.
    """
    root_path = Path(root)
    digest = hashlib.sha256()
    for file_path in discover_files(root_path, **loader_kw):
        rel = file_path.relative_to(root_path).as_posix()
        stat = file_path.stat()
        digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write *data* as JSON to *path* atomically (tmp file + ``os.replace``)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp_path, path)


def save_index(
    retriever: Retriever,
    root: "os.PathLike[str] | str",
    fingerprint: str,
    retriever_name: str,
    engine_version: str,
) -> Path:
    """Persist a fitted *retriever* + metadata for *root*, atomically.

    Writes the retriever's serialized state and its meta file (fingerprint,
    corpus root, creation time, engine version, doc count, retriever name)
    into ``index_dir(root)``, creating the directory if needed; filenames are
    per-retriever (see ``_CACHE_LAYOUT``), so different retrievers' caches
    for the same root coexist. Returns the directory written to.
    """
    data_filename, meta_filename, _cls = _layout(retriever_name)
    directory = index_dir(root)
    directory.mkdir(parents=True, exist_ok=True)

    data = retriever.to_dict()
    _atomic_write_json(directory / data_filename, data)

    units = data.get("units", [])
    file_count = len({u["source_path"] for u in units}) if units else 0
    meta = {
        "fingerprint": fingerprint,
        "corpus_root": Path(root).resolve().as_posix(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_version": engine_version,
        "doc_count": len(data.get("docids", [])),
        "file_count": file_count,
        "retriever_name": retriever_name,
    }
    _atomic_write_json(directory / meta_filename, meta)

    return directory


def load_index(
    root: "os.PathLike[str] | str",
    retriever_name: str = "lexical",
) -> Optional[Tuple[Retriever, Dict[str, Any]]]:
    """Load a previously saved ``(retriever, meta)`` pair for *root*.

    Returns ``None`` if no cache exists for *retriever_name*, or if it
    exists but cannot be parsed/validated (missing files, corrupt JSON,
    missing keys, an unsupported schema version, or a missing optional
    backend needed to deserialize it) — callers should treat ``None`` as
    "no usable cache" and reindex.
    """
    data_filename, meta_filename, retriever_cls = _layout(retriever_name)
    directory = index_dir(root)
    try:
        data = json.loads((directory / data_filename).read_text(encoding="utf-8"))
        meta = json.loads((directory / meta_filename).read_text(encoding="utf-8"))
        retriever = retriever_cls.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return None
    return retriever, meta


def cached_retrievers(root: "os.PathLike[str] | str") -> Dict[str, Dict[str, Any]]:
    """Map of cached retriever name -> meta dict for *root*.

    Only inspects the (small) meta files, so it never triggers a heavyweight
    deserialize; entries whose meta is missing or corrupt are skipped. For
    the shared lexical/lexical+ctx slot, the name comes from the meta's own
    ``retriever_name`` (whichever of the two built the cache last).
    """
    directory = index_dir(root)
    found: Dict[str, Dict[str, Any]] = {}
    seen_meta_files = set()
    for name, (_data_filename, meta_filename, _cls) in _CACHE_LAYOUT.items():
        if meta_filename in seen_meta_files:
            continue
        seen_meta_files.add(meta_filename)
        try:
            meta = json.loads((directory / meta_filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found[meta.get("retriever_name", name)] = meta
    return found


def is_stale(root: "os.PathLike[str] | str", meta: Dict[str, Any], **loader_kw: Any) -> bool:
    """Return True if *root*'s current fingerprint differs from *meta*'s."""
    return compute_fingerprint(root, **loader_kw) != meta.get("fingerprint")
