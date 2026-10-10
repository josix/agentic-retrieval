"""Registry of record-oriented loaders (one ``Document`` per record).

Stdlib-only. A *record loader* turns a whole file into one chunk-granularity
``Document`` per logical record (a gettext entry, a CSV row, a subtitle cue,
...) instead of size-based chunking. The engine core
(``retrieval.project_loader``, ``retrieval.cli``) stays format-agnostic: it only
asks this registry which loader, if any, owns a path's suffix.

Contract for ``RecordLoader.load``: ``load(docid, text, url) -> List[Document]``
where *docid* is the file's relative POSIX path. Each returned ``Document``
uses ``docid=f"{path}:{start}-{end}"``, sets ``source_path``/``start_line``/
``end_line``, and stamps ``metadata["kind"] = loader.kind``. Text-mode CLI hits
are followed by ``detail_lines(metadata)``: either the loader's ``detail_keys``
rendered as ``label: value`` or the output of its optional ``describe`` hook.

Bootstrap: built-in loader modules listed in ``BUILTIN_LOADER_MODULES`` are
imported lazily on first lookup and register themselves at import time, so the
result does not depend on import order. Out-of-tree loaders just call
``register`` at import.

Caveat: ``project_loader.DEFAULT_EXTENSIONS`` is computed when that module is
first imported, so a loader registered later is routed by the chunk loaders but
not discovered by default; pass ``extensions=DEFAULT_EXTENSIONS | {".ext"}``.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from retrieval.document import Document

LoadFn = Callable[[str, str, str], List[Document]]
DescribeFn = Callable[[Mapping[str, Any]], List[Tuple[str, str]]]

BUILTIN_LOADER_MODULES: Tuple[str, ...] = ("retrieval.po_loader",)


@dataclass(frozen=True)
class RecordLoader:
    """A suffix-keyed loader producing one ``Document`` per record."""

    kind: str
    suffixes: FrozenSet[str]
    load: LoadFn
    detail_keys: Tuple[str, ...] = ()
    describe: Optional[DescribeFn] = None


_BY_SUFFIX: Dict[str, RecordLoader] = {}
_BY_KIND: Dict[str, RecordLoader] = {}
_builtins_loaded = False


def _ensure_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True  # set first: re-entrancy guard
    for name in BUILTIN_LOADER_MODULES:
        importlib.import_module(name)


def register(loader: RecordLoader) -> None:
    """Register *loader*; re-registering the same ``kind`` replaces it.

    Raises ``ValueError`` for a suffix without a leading dot or one already
    owned by a different kind.
    """
    suffixes = frozenset(s.lower() for s in loader.suffixes)
    for suffix in suffixes:
        if not suffix.startswith("."):
            raise ValueError(f"record loader suffix must start with '.': {suffix!r}")
        owner = _BY_SUFFIX.get(suffix)
        if owner is not None and owner.kind != loader.kind:
            raise ValueError(f"suffix {suffix!r} already registered by kind {owner.kind!r}")
    unregister(loader.kind)
    _BY_KIND[loader.kind] = loader
    for suffix in suffixes:
        _BY_SUFFIX[suffix] = loader


def unregister(kind: str) -> None:
    """Remove the loader for *kind* and its suffixes (no-op if unknown)."""
    loader = _BY_KIND.pop(kind, None)
    if loader is None:
        return
    for suffix in [s for s, owner in _BY_SUFFIX.items() if owner is loader]:
        del _BY_SUFFIX[suffix]


def lookup(path: str) -> Optional[RecordLoader]:
    """The loader owning *path*'s suffix (case-insensitive), or ``None``."""
    _ensure_builtins()
    return _BY_SUFFIX.get(Path(path).suffix.lower())


def registered_suffixes() -> FrozenSet[str]:
    """All suffixes currently owned by a record loader."""
    _ensure_builtins()
    return frozenset(_BY_SUFFIX)


def one_line(text: str, limit: int = 160) -> str:
    """Flatten newlines to a literal ``\\n`` and truncate to *limit* chars."""
    flat = text.replace("\n", "\\n")
    return flat if len(flat) <= limit else flat[:limit] + "..."


def detail_lines(metadata: Mapping[str, Any]) -> List[str]:
    """Indented ``label: value`` lines for a hit's *metadata* (text mode).

    Empty when there is no metadata or its ``kind`` has no registered loader.
    """
    _ensure_builtins()
    kind = metadata.get("kind") if metadata else None
    loader = _BY_KIND.get(kind) if isinstance(kind, str) else None
    if loader is None:
        return []
    if loader.describe is not None:
        pairs = loader.describe(metadata)
    else:
        pairs = [(k, one_line(str(metadata[k]))) for k in loader.detail_keys if k in metadata]
    return [f"    {label}: {value}" for label, value in pairs]
