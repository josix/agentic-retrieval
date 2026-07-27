# Index and query with the CLI

CLI walkthrough for the `retrieval` console-script: `index`, `query`,
`stats`, the `--force` fast path, `--stale-ok`, `--json`, and the
`/retrieval` subcommands inside Claude Code.

## Sync first

```bash
uv sync --project engine --extra all
```

## `index` — build and persist

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT"
```

With no `--retriever` (the default, `all`), `index` builds every strategy's
cache in one pass — `lexical` (TF-IDF + BM25 fused with RRF, the
zero-dependency baseline), plus `turbovec`, `pi-serini`, and `hybrid`
whenever their optional extras are present — each persisted as its own JSON
cache slot under `<project-root>/.agentic-retrieval` (override with
`RETRIEVAL_INDEX_DIR`, which instead keys a shared base dir by the
project's resolved path). A missing
backend is *skipped*, not a hard failure — the run still exits `0` as long
as `lexical` itself succeeds:

```bash
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT"
# lexical: indexed 42 docs
# turbovec: indexed 42 docs
# pi-serini: indexed 42 docs
# hybrid: indexed 42 docs
# -> /Users/you/project/.agentic-retrieval
```

If, say, `turbovec`'s extras aren't installed, that line reads
`turbovec: skipped (<reason>)` instead — the run still succeeds, and every
other strategy's cache still gets built.

If a fresh (non-stale) cache already exists for a given strategy, `index`
skips that strategy's rebuild and prints `<name>: up to date (use --force to
rebuild)` instead. Pass `--force` to rebuild every strategy unconditionally,
even if fresh:

```bash
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT" --force
```

Pass `--retriever <name>` to build **only** one strategy instead of all of
them — this single-strategy form does not degrade gracefully: a missing
extra hard-fails with exit code 1 (see [full CLI reference](../reference/cli.md)
for exit codes). For example, build the LLM-enriched variant instead of
plain lexical (see [LLM contextualization](llm-contextualization.md) for
cost implications):

```bash
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT" --retriever lexical+ctx
```

## `query` — search the persisted index

```bash
uv run --project engine --extra all retrieval query \
  "what carries data between networks" --root "$PROJECT_ROOT" --top-k 5
```

With no `--retriever` flag (the default, alias `--retriever all`), `query`
loads every available strategy's cache and **consolidates** their rankings
into a single deduplicated, ranked, explainable list (see
[Consolidated query](consolidated-query.md)) instead of picking just one.
Pass `--retriever <name>` (`lexical`, `turbovec`, `pi-serini`, `hybrid`,
`treesitter`) to query a single strategy instead — output stays one
`path:start-end` span per line (best match first, no scores). If that
strategy's cache is missing, or the project's files changed since it was
built (detected by a content fingerprint), `query` auto-reindexes just that
strategy first. Pick `--retriever` per question when you want a single
method — exact keywords -> `lexical`, paraphrase/synonyms -> `turbovec`,
Lucene-grade BM25 -> `pi-serini`, uncertain -> `hybrid` — and fall back to
`--retriever lexical` if the chosen backend was skipped or errors. See
[hybrid fusion](hybrid-fusion.md) for the full routing rationale.

Pass `--stale-ok` to search the existing (possibly stale) cache anyway,
skipping the auto-reindex:

```bash
uv run --project engine --extra all retrieval query "..." --root "$PROJECT_ROOT" --stale-ok
```

Pass `--json` for a machine-readable response instead of one docid per
line:

```bash
uv run --project engine --extra all retrieval query "..." --root "$PROJECT_ROOT" --json
# {"query": "...", "results": ["README.md", "docs/index.md"]}
```

## `stats` — inspect the cache without searching

```bash
uv run --project engine --extra all retrieval stats --root "$PROJECT_ROOT"
```

Prints the cache's root, doc count, creation time, engine version,
staleness, and on-disk location:

```
root: /Users/you/project
docs: 42
created: 2026-07-13T10:00:00Z
engine: <engine version>
stale: False
cache: /Users/you/project/.agentic-retrieval
```

## The `/retrieval` subcommands in Claude Code

Not installed yet? See [Install the plugin](install.md).

```
/retrieval setup
/retrieval index
/retrieval query "your query here"
```

`setup` guards for `uv`, then runs `uv sync --project
"${CLAUDE_PLUGIN_ROOT}/engine" --extra all` and surfaces `uv`'s output
verbatim (retrying a core-only sync first if the full sync fails). `index`
and `query` capture `$CLAUDE_PROJECT_DIR` as the project root before
invoking `uv run` and dispatch to the same `retrieval index`/`retrieval
query` CLI shown above. Full step-by-step protocol lives in
`skills/retrieval/SKILL.md`.

Four knowledge skills auto-trigger for the agent when it needs per-method
depth — you generally won't invoke these yourself, but it's worth knowing
they're there: `lexical-retrieval-usage`, `dense-retrieval-usage`,
`lucene-retrieval-usage`, `hybrid-retrieval-usage`.

## Next steps

- [Full CLI reference](../reference/cli.md) — every flag, exit code, and
  output format.
- [Consolidated query (the default)](consolidated-query.md)
- [Persistence and cache](../reference/persistence-and-cache.md) — how the
  fingerprint and cache directory scheme work.
- [Use each retriever](use-each-retriever.md)
