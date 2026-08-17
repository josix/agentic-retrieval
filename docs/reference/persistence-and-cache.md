# Persistence and cache

On-disk persistence for fitted retrievers, keyed by project path
(`engine/retrieval/persistence.py`). Stdlib-only (`json`, `hashlib`, `os`,
`pathlib`, `time`).

## Cache directory scheme

By default, the cache lives in-project at `<project-root>/.agentic-retrieval`
(`CACHE_DIRNAME`) — resolved from the indexed root's absolute path, so it
follows the project regardless of where it's checked out. This directory is
excluded from file discovery by name, so it never feeds back into an
index/query run.

Set the `RETRIEVAL_INDEX_DIR` environment variable to instead use a shared
base directory outside the project, keyed by a hash of the project path:
`cache_base_dir() / project_key(root)`, where:

- `cache_base_dir()` returns `Path(RETRIEVAL_INDEX_DIR)` when the variable
  is set, else `None` (meaning "use the in-project default" above).
- `project_key(root)` is the SHA-256 of the resolved, absolute POSIX path of
  `root`, truncated to **16 hex characters** — short enough for a directory
  name, long enough that collisions are not a practical concern. Different
  projects (and different checkouts of the same repo, since each resolves
  to a different absolute path) never collide.

Note: caches previously written under the old default,
`~/.cache/agentic-retrieval`, are no longer read or written by this engine
and can be deleted manually.

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
| `doc_count` | `int` | Number of indexed **chunks** (one chunk-Document per span, docid `"{path}:{start}-{end}"`) — despite the name, this is a chunk count, not a file count |
| `file_count` | `int` | Number of distinct source files those chunks came from (`len(set(unit["source_path"] for unit in units))`); usually smaller than `doc_count` once a file yields more than one chunk |
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

The default, in-project cache directory (`.agentic-retrieval`) is excluded
from discovery by name, so the common case (no override) is safe. The
warning below now applies only to overrides.

!!! warning
    If `RETRIEVAL_INDEX_DIR` is pointed *inside* the indexed project root
    using a directory name other than `.agentic-retrieval`, the cache's
    `lexical.json`/`meta.json` get swept up as documents on the next
    index/query run (a feedback loop) — only `.agentic-retrieval` is
    excluded by default (see
    [customize indexing](../how-to/customize-indexing.md)), so any other
    cache dirname is fair game for re-indexing. Point `RETRIEVAL_INDEX_DIR`
    outside the project root instead.

## PDF sidecars + extraction manifest live under the project root

`retrieval.extractors` writes sidecar transcripts and their manifest for
every `extractors.EXTRACTABLE_EXTENSIONS` file (PDFs, plus agent-only media
— `.docx`, `.pptx`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`)
under `<project-root>/.agentic-retrieval/extracted/` — deliberately
**always** under the indexed root, even when `RETRIEVAL_INDEX_DIR` redirects
the retriever caches described above to a shared external directory. A
sidecar is a citation target a coding agent `Read()`s by project-relative
path (`docs/paper.pdf` indexes as
`.agentic-retrieval/extracted/docs/paper.pdf.md`), so it has to live inside
the tree being indexed regardless of where the retriever cache itself is
kept.

The extraction cache (`.agentic-retrieval/extracted/manifest.json`) is keyed
on each source file's SHA-256 content hash plus the extractor's own version
string (`extractors.EXTRACTOR_VERSION`) — a content change or an extractor
upgrade both force re-extraction; an unchanged file across repeated calls
(e.g. `index --auto`'s multiple loader passes in one run) is a cache hit.
See [Customize indexing](../how-to/customize-indexing.md#pdf-auto-indexing)
and the [`extractors` API reference](api/extractors.md).

### Agent-authored manifest entries

A sidecar registered via `retrieval sidecar --register` (see
[CLI reference](cli.md#sidecar)) writes `extractor_version:
"agent-authored/1"` instead of `pypdf-text/1`, plus three extra keys not
present on a pypdf-written entry: `authored_by` (`"agent"`), `authored_at`
(ISO-8601 UTC timestamp), and `sidecar_sha256` (the SHA-256 of the written
sidecar file's own bytes). Both extractor-version strings are accepted as
"fresh" by the cache-hit check, so an agent-authored entry is a durable,
first-class sidecar — not a stub awaiting a real extraction — and is never
silently superseded by a later `pypdf` install; only an explicit `retrieval
extract --force` overwrites it (with a `warning: overwriting N
agent-authored sidecar(s)` line on stderr).

`compute_fingerprint` mixes `sidecar_sha256` into a discovered PDF's
fingerprint line (`"relpath|size|mtime|sidecar_sha256"` instead of the
usual `"relpath|size|mtime"`) whenever that PDF's manifest entry is
agent-authored — so re-registering a changed transcript over an
otherwise-unchanged source PDF (same size/mtime) still flips the
fingerprint and triggers a reindex. A corpus with no agent-authored entries
fingerprints byte-identically to the pre-0.9.0 format.

## Schema versioning

Each persistable retriever carries its own `SCHEMA_VERSION`, bumped
whenever its persisted dict shape changes incompatibly. `from_dict()`
raises `ValueError` on an unrecognized schema version; `load_index`
catches this (along with `OSError`, `json.JSONDecodeError`, `KeyError`,
and the guidance `RuntimeError` raised when an optional backend needed to
deserialize is missing) and returns `None`, so callers reindex from
scratch rather than risk mis-parsing an incompatible on-disk cache.

**v1 -> v2**: every retriever's `SCHEMA_VERSION` bumped from `1` to `2`
when chunk-span metadata (`units`) was added to the persisted dict. A v1
cache on disk (missing `units`, `schema: 1`) is treated exactly like any
other unrecognized-schema cache: `from_dict` raises `ValueError`,
`load_index` returns `None`, and the caller (`_build_and_save` via
`_load_or_rebuild`/`_up_to_date_message`) transparently rebuilds a fresh v2
cache — no manual cache-clearing step is required after upgrading.

## Next steps

- [CLI reference](cli.md)
- [Environment variables](environment-variables.md)
- [Reference: persistence internals](api/retrievers.md)
