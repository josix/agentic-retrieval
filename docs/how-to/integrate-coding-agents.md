# Integrate non-Claude coding agents

Every skill in this repo is plain Markdown with YAML frontmatter — readable
as a reference doc by any agent (or human), even without Claude Code's
plugin machinery. To use the engine outside Claude Code, sync the
environment with `uv` and run the `retrieval` console-script CLI directly.
This is a `uv`-only path — there is no no-`uv` fallback.

## CLI one-liners

```bash
uv sync --project engine --extra all
uv run --project engine --extra all retrieval index --root <path-to-project>
uv run --project engine --extra all retrieval query "..." --root <path-to-project> --top-k 5
```

`index` builds and persists a JSON cache under
`~/.cache/agentic-retrieval/indexes` (override with
`RETRIEVAL_INDEX_DIR`); `query` loads it, auto-reindexing if the cache is
missing or the project's files changed since it was built. `retrieval stats
--root <path-to-project>` reports on the cache without searching.

`query`'s plain output is one `path:start_line-end_line` span per line
(best match first) — feed that straight to your agent's file-read tool:

```bash
uv run --project engine --extra all retrieval query "..." --root <path-to-project> --top-k 1
# -> src/app.py:42-58
```

Read exactly that span — no re-grepping the file to find the right lines:

```
Read(path="src/app.py", offset=42, limit=58 - 42 + 1)
```

`--json` emits `{"query": ..., "results": [{"docid", "path", "start_line",
"end_line", "rank"}, ...]}` for programmatic consumption (see
[CLI reference](../reference/cli.md#query)).

!!! warning
    Don't point `RETRIEVAL_INDEX_DIR` inside the project root being
    indexed — the cache's `lexical.json`/`meta.json` would get indexed as
    documents on the next run (only a `.cache`-named directory is excluded
    by default), creating a feedback loop.

## Heredoc: in-memory engine API

Or drop straight to the in-memory engine API (no persisted cache, rebuilt
per invocation):

```bash
uv sync --project engine --extra all
RETRIEVAL_ROOT=<path-to-project> uv run --project engine --extra all python - <<'PY'
import os

from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import LexicalRetriever

docs = load_chunk_documents(os.environ["RETRIEVAL_ROOT"])
r = LexicalRetriever()
r.index(docs)
for hit in r.search_detailed("...", top_k=5):
    print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}")
PY
```

Or drive the `retrieval.retrievers` API directly — see
[Use each retriever](use-each-retriever.md).

## Environment variable contract

Two environment variables the skills read, if your agent harness sets them:

- `CLAUDE_PLUGIN_ROOT` — the plugin's own directory (where `engine/`
  lives).
- `CLAUDE_PROJECT_DIR` — the project to index (captured into
  `RETRIEVAL_ROOT` before invoking `uv run`).

Neither is required for the manual bootstrap above — pass/set
`RETRIEVAL_ROOT` explicitly instead. Full variable list:
[environment variables reference](../reference/environment-variables.md).

## Next steps

- [Build and release](build-and-release.md)
- [Reference: CLI](../reference/cli.md)
