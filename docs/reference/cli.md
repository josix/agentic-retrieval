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

retrieval query QUERY [--root ROOT] [--retriever lexical|lexical+ctx|turbovec|pi-serini|hybrid]
                       [--top-k N] [--json] [--stale-ok]

retrieval stats [--root ROOT]
```

## `index`

Build and persist an index for a project root.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--root ROOT` | `RETRIEVAL_ROOT` env, then cwd | Project root to index |
| `--retriever {all,lexical,lexical+ctx,turbovec,pi-serini,hybrid}` | **`all`** | Retriever(s) to build |
| `--force` | off | Rebuild even if a fresh (non-stale) cache already exists |

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
-> /home/user/.cache/agentic-retrieval/indexes/<project-key>
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
| `--retriever {lexical,lexical+ctx,turbovec,pi-serini,hybrid}` | `lexical` | Retriever to use if the index needs (re)building |
| `--top-k N` | `5` | Number of results |
| `--json` | off | Emit `{"query": ..., "results": [{"docid", "path", "start_line", "end_line", "rank"}, ...]}` instead of one `path:start-end` span per line |
| `--stale-ok` | off | Search the cached index even if it's stale, instead of auto-reindexing |

If the cache is missing, or present but stale (and `--stale-ok` is not
passed), `query` auto-reindexes (equivalent to a single-strategy `index`)
before searching. Which `--retriever` to query is entirely the caller's
choice — after a default `index all` run has populated every available
cache slot, the coding agent decides per question (exact tokens ->
`lexical`, paraphrase/synonyms -> `turbovec`, Lucene-grade BM25 ->
`pi-serini`, uncertain -> `hybrid`; see `docs/how-to/hybrid-fusion.md`).
`query` never silently falls back to a different retriever: if the chosen
backend's extras are missing, it hard-fails with exit code 1 and a guidance
message on stderr, same as single-strategy `index` — retry with
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

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. Also returned by a default (`all`) `index` run even when one or more optional strategies were skipped for missing extras, as long as `lexical` itself built. |
| `1` | A handled runtime error (e.g. an exception raised inside a subcommand) — the message is printed to stderr as `error: <exc>`. This includes single-strategy `index --retriever <name>` and `query --retriever <name>` when that strategy's extras are missing (no graceful skip in single-strategy form), and a default `index` run whose `lexical` strategy itself fails to build. |
| `2` | Argument-parsing error (missing/invalid flags, unknown subcommand) — raised by `argparse` itself via `SystemExit`, before the CLI's own error handling runs |

## Output formats

- `index`: a single status line to stdout (fast-path or rebuilt message).
- `query` (default): one `source_path:start_line-end_line` span per line,
  best match first, no scores. Feed a span straight to
  `Read(path, offset=start_line, limit=end_line-start_line+1)`.
- `query --json`: a single JSON object, `{"query": "<text>", "results":
  [{"docid": "<path:start-end>", "path": "<source_path>", "start_line":
  <int>, "end_line": <int>, "rank": <int>}, ...]}`. **Breaking change from
  0.2.0**: `results` used to be a flat list of docid strings; it is now a
  list of objects — see [changelog](../changelog.md).
- `stats`: a fixed set of `key: value` lines to stdout.

## `--root` resolution order

`--root` -> `RETRIEVAL_ROOT` environment variable -> current working
directory, if neither is given.

## Next steps

- [Persistence and cache](persistence-and-cache.md)
- [Environment variables](environment-variables.md)
- [Index and query how-to](../how-to/index-and-query.md)
