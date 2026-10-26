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

There are **three distinct index-lifetime and access paths** in this plugin,
and they are not interchangeable:

- **The `retrieval` CLI persists an on-disk cache.** `retrieval index` /
  `retrieval query` build each strategy and persist one cache slot per
  strategy (a data JSON + meta JSON pair; `lexical` and `lexical+ctx` share
  `lexical.json` + `meta.json`) under `<project-root>/.agentic-retrieval`
  (or `$RETRIEVAL_INDEX_DIR/<16-hex project key>` when that variable is
  set). These caches survive across invocations — `query` loads them
  instead of rebuilding, auto-reindexing only when the project's files have
  changed. See [persistence and cache](../reference/persistence-and-cache.md)
  for the full scheme.
- **The heredoc / engine-API path is in-memory, per invocation.** Every
  `load_documents(root)` -> `Retriever().index(docs)` snippet (the
  `python - <<'PY' ... PY` examples throughout this documentation) rebuilds
  the index from scratch in the Python process's memory and discards it on
  exit — nothing gets written to disk, and nothing gets written into your
  project directory. Two consecutive runs of the same heredoc snippet each
  pay the full indexing cost again.
- **The MCP server is a long-lived stdio process per project.** `retrieval
  mcp` (registered by the plugin's `mcpServers` entry and launched through
  `scripts/retrieval-mcp.sh`) reads and writes the **same on-disk caches** as
  the CLI. `lexical` builds lazily on first search; the other strategies are
  built through its `index_project` tool. See
  [MCP server](../how-to/integrate-coding-agents.md#mcp-server).

When choosing between them: use the CLI (`retrieval index`/`query`) for
repeated interactive use where you want the on-disk cache and staleness
detection; use the MCP server for agent sessions that should query the same
cache through tools; use the heredoc/engine-API path for one-off scripting,
tests, or when you need direct access to the `Retriever` object.

## Engine layout

```
.claude-plugin/plugin.json            plugin manifest (name: agentic-retrieval; registers the retrieval MCP server via mcpServers)
.claude-plugin/marketplace.json       single-plugin marketplace entry
commands/retrieval.md                 thin dispatcher -> skill
agents/retrieval-tracer.md            agent: deep code-path tracer for fan-out
scripts/retrieval-mcp.sh              MCP launcher (uv run ... --extra mcp retrieval mcp)
skills/retrieval/SKILL.md             full invocation protocol
skills/lexical-retrieval-usage/       knowledge skill: contextual lexical retrieval
skills/dense-retrieval-usage/         knowledge skill: turbovec dense ANN
skills/lucene-retrieval-usage/        knowledge skill: pi-serini Lucene BM25
skills/code-retrieval-usage/          knowledge skill: tree-sitter AST code retrieval
skills/hybrid-retrieval-usage/        knowledge skill: fusion + method selection
skills/domain-context-retrieval/      skill: ground work in the repo via the MCP tools
engine/                               vendored offline-first retrieval engine (agentic-retrieval package)
  pyproject.toml
  uv.lock
  retrieval/  tests/
    retrieval/__init__.py               re-exports the public API
    retrieval/__main__.py               python -m retrieval -> CLI
    retrieval/retrievers.py             Retriever protocol; Lexical, ContextualLexical, Turbovec, PiSerini, Hybrid, TreeSitter retrievers; REGISTRY, build_retriever
    retrieval/document.py               Document record shared by the retrievers
    retrieval/project_loader.py         discover_files/load_documents/load_chunk_documents over a project root
    retrieval/record_loaders.py         registry of one-Document-per-record loaders
    retrieval/po_loader.py              per-entry gettext .po loader
    retrieval/extractors.py             sidecar-transcript extraction for PDF and agent-only media
    retrieval/chunker.py                Chunk, ChunkingPolicy, line/heading chunker
    retrieval/ast_chunker.py            tree-sitter AST-boundary code chunking
    retrieval/tfidf.py                  sparse TF-IDF and the plain/code/cjk tokenizers
    retrieval/bm25.py                   BM25 Okapi inverted index
    retrieval/index.py                  ContextualRetriever: TF-IDF + BM25 + RRF over chunks
    retrieval/fusion.py                 reciprocal-rank fusion, candidate_pool
    retrieval/consolidation.py          merge per-retriever hits into one deduplicated, explainable ranking
    retrieval/contextualizer.py         offline heuristic breadcrumb contextualizer
    retrieval/llm_contextualizer.py     LLM contextualizer
    retrieval/providers.py              contextualizer/embedder selection
    retrieval/persistence.py            on-disk cache: save_index/load_index/is_stale/cache_is_loadable
    retrieval/autotune.py               signal-based hyperparameter heuristics (retrieval tune / --auto)
    retrieval/eval.py                   labeled-query eval harness (recall, nDCG, latency)
    retrieval/mcp_server.py             stdio MCP server (retrieval mcp)
    retrieval/cli.py                    the retrieval console-script CLI (index/query/mcp/stats/tune/eval/extract/sidecar)
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
- [MCP server](../how-to/integrate-coding-agents.md#mcp-server)
