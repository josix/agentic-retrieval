# CLI reference

The `retrieval` console script, registered via `[project.scripts]` in
`engine/pyproject.toml` (`retrieval = "retrieval.cli:main"`). Thin argparse
dispatcher over `retrieval.persistence` + `retrieval.retrievers`.

## Synopsis

```
retrieval --version

retrieval index [--root ROOT] [--retriever all|lexical|lexical+ctx|turbovec|pi-serini|hybrid]
                [--force]
                # --retriever defaults to "all"

retrieval query QUERY [--root ROOT]
                       [--retriever all|lexical|lexical+ctx|turbovec|pi-serini|hybrid|treesitter]
                       [--top-k N] [--json] [--stale-ok]
                       [--weights "name:w,..."] [--output PATH]
                       # --retriever defaults to "all" (consolidated mode)

retrieval stats [--root ROOT]

retrieval eval --queries PATH [--root PATH] [--k 5] [--warm-runs 5]
               [--json] [--output PATH]

retrieval extract [--root ROOT] [--force] [--prune] [--json]

retrieval sidecar [--root ROOT] [--json] (--list | --register SOURCE --transcript PATH|-)
```

## `index`

Build and persist an index for a project root.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--root ROOT` | `RETRIEVAL_ROOT` env, then cwd | Project root to index |
| `--retriever {all,lexical,lexical+ctx,turbovec,pi-serini,hybrid,treesitter}` | **`all`** | Retriever(s) to build |
| `--force` | off | Rebuild even if a fresh (non-stale) cache already exists |
| `--no-pdf` | off | Exclude PDFs and every other sidecar-routed media suffix (docx/pptx/xlsx/images) from discovery — an escape hatch for the default auto-activated sidecar-extraction pipeline (see [Customize indexing](../how-to/customize-indexing.md#pdf-auto-indexing)). Persisted in the cache's meta, so it's sticky across later flag-less `index`/`query` calls |
| `--allow-large-context` | off | Skip the `lexical+ctx` large-corpus guard (warns above 500 chunks, refuses above 2000 without this flag — each chunk costs one LLM call at index time) |

Each retriever has its own cache slot per project, so e.g. a fresh
`lexical` cache never short-circuits a `hybrid` build.

Indexing is **chunk**-granularity: `index` chunks every discovered file
(`retrieval.project_loader.load_chunk_documents`) and persists one entry
per chunk span, docid `"{path}:{start}-{end}"`. This is what lets `query`
return file:line spans instead of whole-file docids.

### Default (`all`) — build every strategy in one pass

With no `--retriever` (or `--retriever all`), `index` builds each strategy in
`lexical`, `turbovec`, `pi-serini`, `hybrid` — deliberately excluding
`lexical+ctx`, since it shares the `lexical` cache slot and costs LLM tokens.
For each strategy, in order: if a fresh cache exists and `--force` wasn't
passed, prints `<name>: up to date (use --force to rebuild)` and moves on;
otherwise it builds and persists that strategy, printing
`<name>: indexed <N> chunks`. If a strategy's optional extras are missing, it
prints `<name>: skipped (<reason>)` **instead of failing** — this is the
"graceful degradation" mode: the whole run still exits `0` as long as the
always-available `lexical` strategy itself built (or was already fresh).
A final `-> <cache-dir>` line is printed. Sample output when `turbovec` is
uninstalled:

```
lexical: indexed 118 chunks
turbovec: skipped (turbovec retriever needs the 'turbovec' + 'local' extras:)
pi-serini: indexed 118 chunks
hybrid: skipped (turbovec retriever needs the 'turbovec' + 'local' extras:)
-> /home/user/project/.agentic-retrieval
```

The skipped `<reason>` is the first line of the backend's guidance
`RuntimeError`; run the single-strategy form (`--retriever <name>`) to see
the full message including the exact install command.

### Single strategy (`--retriever <name>`)

Pass an explicit `--retriever lexical|lexical+ctx|turbovec|pi-serini|hybrid`
to build just one strategy. This form does **not** degrade gracefully:
without `--force`, if a fresh cache already exists for the chosen retriever,
`index` skips the rebuild and prints `<retriever> index up to date -> <dir>
(use --force to rebuild)`, returning exit code 0. Otherwise it builds the
retriever over `load_chunk_documents(root)`, persists it, and prints
`indexed <N> chunks -> <dir>  fingerprint=<12-char prefix>`.

Optional extras per retriever: `lexical` needs none; `lexical+ctx` needs the
contextualizer's own dependencies; `turbovec` and `hybrid` need the
`turbovec` + `local` extras; `pi-serini` needs the `pyserini` extra plus a
Java 21 runtime. In this single-strategy form, a missing extra surfaces as
exit code **1** with a guidance message on stderr — no skip, no fallback.

## `query`

Search the persisted index for a project root.

| Flag | Default | Purpose |
| --- | --- | --- |
| `query` (positional) | — | Query text (required) |
| `--root ROOT` | `RETRIEVAL_ROOT` env, then cwd | Project root to search |
| `--retriever {all,lexical,lexical+ctx,turbovec,pi-serini,hybrid,treesitter}` | **`all`** | `all` (default) consolidates every available strategy; a single name queries just that strategy |
| `--top-k N` | `5` | Number of results |
| `--json` | off | Emit JSON instead of plain text (shape depends on `--retriever`; see "Output formats" below) |
| `--stale-ok` | off | Search the cached index even if it's stale, instead of auto-reindexing |
| `--weights "name:w,..."` | none | Consolidated mode only: per-retriever RRF weight override (unlisted retrievers default to `1.0`) |
| `--output PATH` | none | Consolidated mode only: also write the JSON envelope to `PATH` |

### Default (`all`) — consolidated, deduplicated, explainable ranking

With no `--retriever` (or the explicit `--retriever all` alias), `query`
loads (auto-reindexing as needed) every strategy in `lexical`, `turbovec`,
`pi-serini`, `hybrid`, `treesitter`, searches each, and merges/fuses the
results with `retrieval.consolidation.consolidate` into a single
deduplicated, ranked list — see [Consolidated
query](../how-to/consolidated-query.md) for the full output format and the
span-merge/weighted-RRF mechanics. A missing backend's extras are skipped
(reported on stderr in text mode, in the JSON envelope's `"skipped"` list
otherwise) — never a hard failure, as long as `lexical` consolidates
successfully (exit 0); exit 1 only if even `lexical` is unusable.

### Single strategy (`--retriever <name>`)

Pass an explicit `--retriever lexical|lexical+ctx|turbovec|pi-serini|hybrid|treesitter`
to query just one strategy, unchanged from the pre-consolidation CLI: if the
cache is missing, or present but stale (and `--stale-ok` is not passed),
`query` auto-reindexes (equivalent to a single-strategy `index`) before
searching. Pick `--retriever` per question (exact tokens -> `lexical`,
paraphrase/synonyms -> `turbovec`, Lucene-grade BM25 -> `pi-serini`,
uncertain -> `hybrid`, AST-boundary code spans -> `treesitter`; see
`docs/how-to/hybrid-fusion.md`). This form never silently falls back to a
different retriever: if the chosen backend's extras are missing, it
hard-fails with exit code 1 and a guidance message on stderr — retry with
`--retriever lexical` explicitly if you want that fallback.

## `stats`

Report on every persisted index cache for a project root (one block per
cached retriever, blank-line separated), without searching or rebuilding.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--root ROOT` | `RETRIEVAL_ROOT` env, then cwd | Project root to report on |

Prints `retriever`, `root`, `chunks` (chunk count), `files` (distinct
source-file count), `created`, `engine`, `stale`, and `cache` (the on-disk
directory) per cached retriever. If no cache exists, prints
`no cache for <root> (dir=<cache-dir>)` and returns exit code 0.

## `eval`

Run the labeled-query eval harness — recall@k/nDCG@k per retriever vs the
consolidated fusion, confidence-signal validity, and cold/warm search
latency — entirely in-memory (no `<project-root>/.agentic-retrieval`
writes). See
[Evaluate retrievers](../how-to/evaluate-retrievers.md) for the query-set
schema and how to read the metrics.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--queries PATH` | required | Path to a labeled `eval_queries.json` file |
| `--root PATH` | the query set's own `corpus_root` | Override the corpus to eval against |
| `--k N` | `5` | recall@k / nDCG@k cutoff |
| `--warm-runs N` | `5` | Number of extra warm search repeats per query |
| `--json` | off | Emit a JSON report instead of text |
| `--output PATH` | none | Also write the JSON report to `PATH` |

Builds every strategy in `lexical`, `turbovec`, `pi-serini`, `hybrid`,
`treesitter` over the query set's corpus, skipping (not hard-failing) any
whose optional extras are missing — same graceful-degradation convention as
`index`/`query`'s default mode. Exit code `0` as long as `lexical` produced
results; `1` only if even `lexical` is unusable.

## `extract`

Pre-warm every PDF's sidecar transcript under a project root, without
building or touching any retriever index — useful for pre-warming a large
corpus's extraction cache ahead of time (e.g. in CI, or before a first
`index` run) so that `index`/`query` don't pay the extraction cost inline.
**PDF only** — `extract` never attempts agent-only media (`.docx`, `.pptx`,
`.xlsx`, images), which have no machine extractor to run; those always need
`sidecar --register`.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--root ROOT` | `RETRIEVAL_ROOT` env, then cwd | Project root to extract from |
| `--force` | off | Re-extract every PDF even if its cached sidecar is already fresh |
| `--prune` | off | Remove manifest entries and sidecar files for PDFs and media no longer present under `--root` (registered agent-only sidecars are kept, not just PDFs, even though this loop never extracts them) |
| `--json` | off | Emit a JSON summary instead of one progress line per file |

Discovers PDFs the same way `index`/`query` do (`discover_files` with
default kwargs, filtered to `MACHINE_EXTRACTABLE_EXTENSIONS`), then calls
`retrieval.extractors.ensure_sidecar` on each. This is the **only** place
PDF extraction hard-fails when the `pdf` extra isn't installed: a preflight
`require_extractors` call raises a guidance `RuntimeError` (exit code 1,
`error: PDF extraction needs the 'pdf' extra:\n  uv pip install -e
'.[pdf]'` on stderr) before any file is touched — `index`/`query` never do
this; they degrade to backend-missing stubs instead (see [Customize
indexing](../how-to/customize-indexing.md#pdf-auto-indexing)).

Text-mode output is one line per PDF:

```
docs/paper.pdf -> .agentic-retrieval/extracted/docs/paper.pdf.md (12 pages, 34521 chars)
docs/scan.pdf: stub (no-text-layer)
```

`--json` emits `{"root": "<path>", "files": [{"source", "sidecar",
"status", "reason", "pages", "chars"}, ...], "pruned": <int>}`. Exit code
`0` on success even if some PDFs produced stubs (a stub is still a valid,
searchable sidecar) — non-zero exit is reserved for the missing-`pdf`-extra
preflight failure.

## `sidecar`

Inspect or author PDF/media sidecar transcripts without touching pypdf —
the no-install alternative to `extract` for PDFs, and the **only**
indexing path for agent-only media (`.docx`, `.pptx`, `.xlsx`, images),
which have no machine extractor at all. Unlike `extract`, `sidecar` never
imports/requires `pypdf` on either mode, so it works identically whether or
not the `pdf` extra is installed.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--root ROOT` | `RETRIEVAL_ROOT` env, then cwd | Project root |
| `--json` | off | Emit JSON instead of text |
| `--list` | — | Report every discovered PDF/media file's sidecar state (mutually exclusive with `--register`, one required) |
| `--register SOURCE` | — | Register an agent-authored transcript as `SOURCE`'s sidecar (requires `--transcript`; mutually exclusive with `--list`, one required) |
| `--transcript PATH\|-` | required with `--register` | Path to the transcript file, or `-` for stdin |

### `--list`

Reports every PDF/media file `discover_files` would surface, one state per
file: `missing` (no manifest entry yet), `outdated` (source changed since
the entry was written), `agent-authored` (registered via `--register`),
`stub` (a placeholder, e.g. `backend-missing` for a PDF or `agent-only` for
media with no machine extractor), or `ok` (a fresh pypdf transcript,
PDF-only). Text mode: one line per file, e.g.

```
docs/paper.pdf: stub (backend-missing) -> needs transcript
docs/spec.pdf: agent-authored (4 pages)
img/diagram.png: stub (agent-only) -> needs transcript
```

`--json` emits `{"root", "backend_available", "files": [{"rel", "state",
"sidecar", "status", "reason", "pages", "authored_at"}, ...]}`.

### `--register SOURCE --transcript PATH|-`

Writes `SOURCE`'s sidecar from a hand-authored transcript — the recovery
path when an agent has read the PDF (or docx/pptx/xlsx/image) itself (e.g.
via its own `Read` tool, or vision for an image) instead of installing
`pypdf`. See the `retrieval` skill's "Without the `pdf` extra: author the
transcript yourself" section for the full agent workflow (page-heading
format — optional for page-less formats, when to use this, lifecycle).
Prints a `reindex to pick up this sidecar` hint; also warns on stderr if
`SOURCE` isn't discoverable by the default loader, or if the transcript has
no `## Page N` headings (page breadcrumbs will be absent from search-hit
context — harmless for a page-less format). Missing `--transcript` is an
argument-parsing error (exit code 2, mirroring argparse's own
required-argument handling).

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. Also returned by a default (`all`) `index` run, and a default (`all`) `query` run, even when one or more optional strategies were skipped for missing extras, as long as `lexical` itself built/consolidated. |
| `1` | A handled runtime error (e.g. an exception raised inside a subcommand) — the message is printed to stderr as `error: <exc>`. This includes single-strategy `index --retriever <name>` and `query --retriever <name>` when that strategy's extras are missing (no graceful skip in single-strategy form), a default `index` run whose `lexical` strategy itself fails to build, and a default (`all`) `query` run where even `lexical` couldn't be consolidated. |
| `2` | Argument-parsing error (missing/invalid flags, unknown subcommand) — raised by `argparse` itself via `SystemExit`, before the CLI's own error handling runs |

## Output formats

- `index`: a single status line to stdout (fast-path or rebuilt message).
- `query` (default, consolidated `all` mode): one
  `source_path:start_line-end_line  [score=... agree=n/m conf=...  via
  a,b,c]  context` line per result, best match first (skip notes go to
  stderr). The first token stays `path:start-end`, so it still feeds
  straight into `Read(path, offset=start_line,
  limit=end_line-start_line+1)`.
- `query --retriever <name>` (single strategy): one
  `source_path:start_line-end_line` span per line, best match first, no
  scores — unchanged from before consolidated mode existed.
- `query --json` (default, consolidated `all` mode): `{"query": "<text>",
  "mode": "consolidated", "retrievers": [...], "skipped": [{"name",
  "reason"}, ...], "results": [{"docid", "path", "start_line", "end_line",
  "rank", "context", "score", "provenance", "agreement", "confidence",
  "contributors"}, ...]}`.
- `query --retriever <name> --json` (single strategy): `{"query": "<text>",
  "results": [{"docid": "<path:start-end>", "path": "<source_path>",
  "start_line": <int>, "end_line": <int>, "rank": <int>, "context": "<str>"}, ...]}` —
  unchanged from before consolidated mode existed. **Breaking change from
  0.2.0**: `results` used to be a flat list of docid strings; it is now a
  list of objects — see [changelog](../changelog.md).
- `query --output PATH` (consolidated mode only): also writes the
  `--json` envelope to `PATH`.
- `stats`: a fixed set of `key: value` lines to stdout.

## `--root` resolution order

`--root` -> `RETRIEVAL_ROOT` environment variable -> current working
directory, if neither is given.

## Next steps

- [Consolidated query (the default)](../how-to/consolidated-query.md)
- [Persistence and cache](persistence-and-cache.md)
- [Environment variables](environment-variables.md)
- [Index and query how-to](../how-to/index-and-query.md)
