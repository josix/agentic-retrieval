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
`<project-root>/.agentic-retrieval` (override with
`RETRIEVAL_INDEX_DIR`); `query` loads it, auto-reindexing if the cache is
missing or the project's files changed since it was built. `retrieval stats
--root <path-to-project>` reports on the cache without searching.

`query`'s plain output is one `path:start_line-end_line` span per line
(best match first) — feed that straight to your agent's file-read tool:

```bash
uv run --project engine --extra all retrieval query "..." --root <path-to-project> --top-k 1
# -> src/app.py:42-58
```

The span locates the lines for you — open it directly instead of re-grepping the file, then read outward from there to follow references and confirm the answer:

```
Read(path="src/app.py", offset=42, limit=58 - 42 + 1)
```

`--json` emits `{"query": ..., "results": [{"docid", "path", "start_line",
"end_line", "rank"}, ...]}` for programmatic consumption (see
[CLI reference](../reference/cli.md#query)).

!!! warning
    Don't point `RETRIEVAL_INDEX_DIR` inside the project root being
    indexed using a directory name other than the default's
    `.agentic-retrieval` — the cache's `lexical.json`/`meta.json` would get
    indexed as documents on the next run (only `.agentic-retrieval` is
    excluded by default), creating a feedback loop.

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

## Deep answers with non-Claude agents

The same phased deep-answer workflow the `retrieval` skill's Step 3 defines
(Q → R → T → C → S) works with only the CLI — no Claude plugin machinery
required:

1. **Decompose** the question into 3-6 sub-aspects (entry point, data flow,
   core algorithm, edge cases, downstream consumption).
2. **Retrieve** once per sub-aspect: `retrieval query "<sub-question>" --root
   <path-to-project> --json --output <scratch>/aspect-N.json`. Persisting each
   sub-aspect's seeds separately keeps them from being lost mid-trace.
3. **Trace** every seed span with your own agent's read/grep tools: open the
   span in the live file, follow callers/callees/imports/config outward, and
   re-query with vocabulary a hit reveals.
4. **Coverage-check** before answering: every sub-aspect has a verified
   `file:line`, the execution path is traced entry to exit, and every cited
   span has been checked against the live file (not just the cache). Loop
   back to step 2 or 3 if any of that is missing.

## MCP server

`retrieval mcp` serves the engine over MCP stdio (needs the `mcp` extra). The
Claude Code plugin registers it in `.claude-plugin/plugin.json` (`mcpServers`); for other clients:

```bash
claude mcp add retrieval -- uv run --project <engine> --extra mcp retrieval mcp
```

Tools:

- `index_status()` -- per-strategy `indexed` / `loadable` / `stale`, plus a
  `recommendation`.
- `search_domain_context(query, retriever="all", top_k, pool, weights,
  stale_ok, build_missing=False, include_snippet=True, snippet_lines=20)` --
  consolidated, explainable hits (`path`, `start_line`, `end_line`,
  `provenance`, `agreement`, `confidence`) with code snippets, plus `skipped`,
  `stale` and a `hint`. Lexical builds lazily; other strategies join once
  built via `index_project`. `stale_ok=None` (default) refreshes only lexical.
- `index_project(retriever="all", force=False, auto=False)` -- build or
  refresh caches (slow for `all` on large repos).

Tool descriptions tell the agent to search first; `top_k` is clamped to 25,
snippets are byte-capped (`truncated` flag), and errors come back as
`{"error": ...}` / `skipped` entries rather than exceptions. Search results
include an `index` block (`stale`, `created_at`). The server also exposes the
resource `retrieval://index/status` and the prompt `ground_implementation(task)`.
The `domain-context-retrieval` skill drives these tools automatically.

The cache lives in `<root>/.agentic-retrieval` unless `RETRIEVAL_INDEX_DIR`
is set.

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
