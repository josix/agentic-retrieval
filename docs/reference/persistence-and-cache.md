# Persistence and cache

On-disk persistence for fitted retrievers, keyed by project path
(`engine/retrieval/persistence.py`). Stdlib-only (`json`, `hashlib`, `os`,
`pathlib`, `time`).

## Cache directory scheme

The cache lives at `cache_base_dir() / project_key(root)`:

- `cache_base_dir()` defaults to `~/.cache/agentic-retrieval/indexes`
  (`DEFAULT_BASE`); override with the `RETRIEVAL_INDEX_DIR` environment
  variable.
- `project_key(root)` is the SHA-256 of the resolved, absolute POSIX path of
  `root`, truncated to **16 hex characters** — short enough for a directory
  name, long enough that collisions are not a practical concern. Different
  projects (and different checkouts of the same repo, since each resolves
  to a different absolute path) never collide.

Each retriever gets its own data + meta file pair inside the per-project
cache directory, so different retrievers' caches for the same project
coexist:

| Retriever | Data file | Meta file |
|---|---|---|
| `lexical` / `lexical+ctx` | `lexical.json` | `meta.json` |
| `turbovec` | `turbovec.json` | `turbovec.meta.json` |
| `pi-serini` | `pi-serini.json` | `pi-serini.meta.json` |
| `hybrid` | `hybrid.json` | `hybrid.meta.json` |

`lexical` and `lexical+ctx` share one slot because a ctx index's persisted
state is plain `LexicalRetriever` data (the LLM context is baked into the
indexed text at build time); the meta's `retriever_name` records which of
the two built it last. `pi-serini` additionally keeps its binary Lucene
segments in a `lucene/` subdirectory of the same cache dir — its JSON file
only points at them. `turbovec` (and `hybrid`'s dense arm) persists docids
plus the raw embedding vectors and rebuilds the quantized ANN index on load.
`cached_retrievers(root)` returns a `{name: meta}` map of every parsable
cache slot for a root.

A single default `retrieval index` run (no `--retriever`, i.e. `all`)
populates several of these independent slots in one pass — `lexical`,
`turbovec`, `pi-serini`, and `hybrid` (whichever backends' extras are
present) — each with its own `fingerprint`/`created_at`/staleness, so e.g.
the `turbovec` slot can go stale and get rebuilt independently of the
`lexical` slot.

## `meta.json` fields

| Field | Type | Meaning |
|---|---|---|
| `fingerprint` | `str` | SHA-256 fingerprint of the corpus at index time (see below) |
| `corpus_root` | `str` | Resolved, absolute POSIX path of the indexed root |
| `created_at` | `str` | UTC timestamp, `%Y-%m-%dT%H:%M:%SZ` |
| `engine_version` | `str` | `retrieval.__version__` at index time |
| `doc_count` | `int` | Number of documents in the index |
| `retriever_name` | `str` | The `--retriever` value used to build this cache |

## Fingerprint

The fingerprint is a SHA-256 hash computed by `compute_fingerprint(root)`
over every file `discover_files(root)` returns, in deterministic sorted
order, one line per file:

```
<relpath>|<size>|<mtime_ns>\n
```

— `relpath` is the file's POSIX-style path relative to `root`, `size` is
`st_size`, and `mtime_ns` is `st_mtime_ns`. Any file added, removed,
resized, or touched (mtime change) alters the fingerprint, without reading
file contents. `is_stale(root, meta)` recomputes the current fingerprint
and compares it against `meta["fingerprint"]`.

## Atomic writes

Every cache JSON file (data and meta alike) is written atomically:
`_atomic_write_json` writes to a `.tmp` sibling file, then calls
`os.replace()` to move it into place. A crash mid-write never leaves a
half-written cache file for a future `load_index` to trip over — a
corrupt/incomplete JSON file causes `load_index` to return `None` (treated
as "no usable cache"), never a partially-written one.

## `--force` / `--stale-ok` semantics

- **`index --force`**: rebuild unconditionally, even if a fresh (non-stale)
  cache already exists. Without `--force`, `index` checks for an existing,
  non-stale cache first and skips the rebuild if found (the "fast path").
- **`query --stale-ok`**: search the existing cached index even if it's
  stale (fingerprint mismatch), instead of auto-reindexing first. Without
  `--stale-ok`, `query` rebuilds automatically whenever the cache is
  missing or stale.

## `RETRIEVAL_INDEX_DIR` + self-indexing warning

!!! warning
    If `RETRIEVAL_INDEX_DIR` is pointed *inside* the indexed project root,
    the cache's `lexical.json`/`meta.json` get swept up as documents on the
    next index/query run (a feedback loop) — only a directory literally
    named `.cache` is excluded by default (see
    [customize indexing](../how-to/customize-indexing.md)), so any other
    cache dirname is fair game for re-indexing. Point `RETRIEVAL_INDEX_DIR`
    outside the project root instead.

## Schema versioning

Each persistable retriever carries its own `SCHEMA_VERSION`, bumped
whenever its persisted dict shape changes incompatibly. `from_dict()`
raises `ValueError` on an unrecognized schema version; `load_index`
catches this (along with `OSError`, `json.JSONDecodeError`, `KeyError`,
and the guidance `RuntimeError` raised when an optional backend needed to
deserialize is missing) and returns `None`, so callers reindex from
scratch rather than risk mis-parsing an incompatible on-disk cache.

## Next steps

- [CLI reference](cli.md)
- [Environment variables](environment-variables.md)
- [Reference: persistence internals](api/retrievers.md)
