# Architecture

## uv-managed, per-plugin environment

The environment is **`uv`-managed and per-plugin, not per-project** —
`uv sync --project engine`/`uv run --project engine` resolve to a single
environment shared across every project root you point `load_documents`/
`load_chunks` at, by default under `uv`'s own cache. Set
`UV_PROJECT_ENVIRONMENT` to pin it to a specific path (e.g.
`$HOME/.cache/agentic-retrieval/uv-venv`, what the plugin path in
`skills/retrieval/SKILL.md` uses) — useful when the plugin's cache
directory isn't writable. Everything is synced once, not reinstalled per
project.

Always invoke the engine via `uv run --project <path-to-engine>` (or
`uv run --directory engine` from inside a checkout) — never a bare
`python`/`python3` call, which resolves to a different, dependency-less
interpreter.

## Where the index lives

There are **two distinct index-lifetime models** in this plugin, and they
are not interchangeable:

- **The `retrieval` CLI persists an on-disk cache.** `retrieval index` /
  `retrieval query` build a `LexicalRetriever` and persist it as a JSON
  cache (`lexical.json` + `meta.json`) under
  `~/.cache/agentic-retrieval/indexes/<project-key>` (override with
  `RETRIEVAL_INDEX_DIR`). This cache survives across invocations —
  `query` loads it instead of rebuilding, auto-reindexing only when the
  project's files have changed. See
  [persistence and cache](../reference/persistence-and-cache.md) for the
  full scheme.
- **The heredoc / engine-API path is in-memory, per invocation.** Every
  `load_documents(root)` -> `Retriever().index(docs)` snippet (the
  `python - <<'PY' ... PY` examples throughout this documentation) rebuilds
  the index from scratch in the Python process's memory and discards it on
  exit — nothing gets written to disk, and nothing gets written into your
  project directory. Two consecutive runs of the same heredoc snippet each
  pay the full indexing cost again.

When choosing between the two: use the CLI (`retrieval index`/`query`) for
repeated interactive use where you want the on-disk cache and staleness
detection; use the heredoc/engine-API path for one-off scripting, tests, or
when you need direct access to the `Retriever` object.

## Engine layout

```
.claude-plugin/plugin.json            plugin manifest (name: agentic-retrieval)
.claude-plugin/marketplace.json       single-plugin marketplace entry
commands/retrieval.md                 thin dispatcher -> skill
skills/retrieval/SKILL.md             full invocation protocol
skills/lexical-retrieval-usage/       knowledge skill: contextual lexical retrieval
skills/dense-retrieval-usage/         knowledge skill: turbovec dense ANN
skills/lucene-retrieval-usage/        knowledge skill: pi-serini Lucene BM25
skills/hybrid-retrieval-usage/        knowledge skill: fusion + method selection
engine/                               vendored offline-first retrieval engine (agentic-retrieval package)
  pyproject.toml
  uv.lock
  retrieval/  tests/
    retrieval/retrievers.py             LexicalRetriever, ContextualLexicalRetriever, TurbovecRetriever, PiSeriniRetriever, REGISTRY, build_retriever
    retrieval/document.py               Document record shared by the retrievers
    retrieval/project_loader.py         discover_files/load_documents/load_chunks over a project root
    retrieval/persistence.py            on-disk cache: save_index/load_index/is_stale
    retrieval/cli.py                    the retrieval console-script CLI
    tests/fixtures/                     engineered fixtures for the Routing-chunk contextualization test
```

## Note on engine provenance

All active engine code lives under `engine/`, vendored from
`contextual-retrieval-exp`. (A legacy `reference` symlink to a sibling
directory used to sit at the repo root; it has been removed.)

## Cache caveats

Because plugins run from `~/.claude/plugins/cache`, any path *outside* the
plugin directory won't resolve once installed — keep every path the plugin
reads at runtime inside the plugin directory itself.

## Next steps

- [Persistence and cache reference](../reference/persistence-and-cache.md)
- [Engine API reference](../reference/engine-api.md)
