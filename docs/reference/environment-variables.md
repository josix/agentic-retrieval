# Environment variables

| Variable | Read by | Default | Purpose |
|---|---|---|---|
| `RETRIEVAL_ROOT` | `retrieval.cli._resolve_root` | (none) | Project root to index/query/report on, when `--root` isn't passed |
| `PROJECT_ROOT` | shell convention (skills/how-to snippets) | (none) | Not read by the engine directly — a shell variable convention used to capture `$CLAUDE_PROJECT_DIR`/`$(pwd)` before invoking `uv run`, then passed as `--root` or exported as `RETRIEVAL_ROOT` |
| `CLAUDE_PROJECT_DIR` | Claude Code (set by the harness) | (none) | The invoking project's directory — captured into `PROJECT_ROOT`/`RETRIEVAL_ROOT` before `uv run`, per the skills' convention |
| `CLAUDE_PLUGIN_ROOT` | Claude Code (set by the harness) | (none) | The plugin's own directory (where `engine/` lives when installed) — used as the `--project` path for `uv sync`/`uv run` |
| `UV_PROJECT_ENVIRONMENT` | `uv` itself | (uv's own cache) | Pins the engine's `uv`-managed virtualenv to a specific path — the skills default this to `$HOME/.cache/agentic-retrieval/uv-venv`, useful when the plugin's cache directory isn't writable |
| `RETRIEVAL_INDEX_DIR` | `retrieval.persistence.cache_base_dir` | (none — `index_dir` falls back to `<project-root>/.agentic-retrieval`) | Relocates the on-disk index cache to a shared `<override>/<project-key>` base directory — see [persistence and cache](persistence-and-cache.md) for the self-indexing feedback-loop warning before pointing it inside a project root |
| `ANTHROPIC_API_KEY` | `retrieval.llm_contextualizer._new_anthropic_client` | (none) | Required for real LLM contextualization (`ContextualLexicalRetriever`, `LLMContextualizer`, `contextualize_llm`) — missing key raises a guidance `RuntimeError` |
| `RAG_PROVIDER` | `retrieval.providers.get_contextualizer` | `heuristic` | Selects a provider stub: `heuristic` (default, offline, working), or `ollama`/`remote`/`sentence_transformers` (currently stubs that always raise `RuntimeError` on use) |

## Per-variable notes

### `RETRIEVAL_ROOT`

Resolution order for every CLI subcommand is `--root` -> `RETRIEVAL_ROOT`
-> current working directory. Not read by the engine-API heredoc snippets
directly — those snippets read it themselves via
`os.environ["RETRIEVAL_ROOT"]` as a documentation convention, not an
engine-level default.

### `PROJECT_ROOT` / `CLAUDE_PROJECT_DIR`

`PROJECT_ROOT` is not read by any engine code — it's a shell-level
convention used throughout the how-to pages and skills to capture the
project root **before** invoking `uv run` (since `uv run --project engine`
changes what the *engine* resolves relative to, not your shell's cwd):

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
```

### `CLAUDE_PLUGIN_ROOT`

Only meaningful when running as an installed Claude Code plugin, where
Claude Code runs the plugin from `~/.claude/plugins/cache` rather than a
working copy on disk — so `${CLAUDE_PLUGIN_ROOT}/engine` resolves inside
that cache directory. Not required for the standalone/manual bootstrap
(see [integrate coding agents](../how-to/integrate-coding-agents.md)).

### `UV_PROJECT_ENVIRONMENT`

Set by the plugin's skill scripts to
`$HOME/.cache/agentic-retrieval/uv-venv` by default; override to pin the
environment elsewhere. Not required outside the plugin context — a plain
`uv sync --project engine` without this set still works, just using `uv`'s
own default virtualenv cache location.

### `RETRIEVAL_INDEX_DIR`

See [persistence and cache](persistence-and-cache.md) for the full cache
directory scheme and the self-indexing feedback-loop warning.

### `ANTHROPIC_API_KEY`

Only needed for the `remote` extra's LLM contextualization paths — the
default heuristic contextualizer needs no key. See
[LLM contextualization](../how-to/llm-contextualization.md).

### `RAG_PROVIDER`

Only `heuristic` (the default) is a working provider. Setting it to
`ollama`, `remote`, or `sentence_transformers` selects a stub that always
raises `RuntimeError` when actually used — these are wiring points for
future providers, not functioning integrations. See
[LLM contextualization](../how-to/llm-contextualization.md#rag_provider-stubs).

## Next steps

- [CLI reference](cli.md)
- [Persistence and cache](persistence-and-cache.md)
