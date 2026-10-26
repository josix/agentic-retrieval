# Troubleshooting

## Engine / CLI

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: turbovec retriever needs the 'turbovec' + 'local' extras` | `turbovec`+`local` extras not synced | `uv sync --project engine --extra all` |
| `RuntimeError: pi-serini retriever needs the 'pyserini' extra and Java 21` | `pyserini` extra not synced, or sync succeeded but no Java 21 on `PATH` | Install Java 21, then `uv sync --project engine --extra all`; verify `java -version` reports 21 |
| `RuntimeError: LLM contextualizer needs the 'anthropic' package + ANTHROPIC_API_KEY` | `remote` extra not synced, or key not exported | `uv sync --project engine --extra all` and `export ANTHROPIC_API_KEY=...` |
| `load_documents(...)` returns `[]` | Root path wrong, or everything under it is excluded (empty dir, all `.git`/`.venv`/etc.) | Point the root at the real project directory; loosen exclusions per [customize indexing](../how-to/customize-indexing.md) if needed |
| `RuntimeError: call index() before search()` | Called `.search()` before `.index()`/`.build()` succeeded | Always index first; if index failed with a `RuntimeError`, fix that first |
| `uv: command not found` | `uv` isn't installed | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| PDF search results are placeholder stubs (`warning: pypdf is not installed`) | `pdf` extra not synced — the expected first state under the skill's default sync, which delegates media extraction to the agent | Author the transcript yourself: `retrieval sidecar --register <pdf> --transcript <file> --root <root>` (no install required — see the `retrieval` skill's "Author the media transcript yourself (the default media path)"). Alternatively opt in to machine extraction: `uv sync --project engine --extra pdf` and reindex |
| docx/pptx/xlsx/image search results are placeholder stubs (`warning: media files were indexed as agent-transcribable stubs`) | Expected — these formats have **no** machine extractor at all, `pdf` extra or not | Author the transcript yourself: `retrieval sidecar --register <file> --transcript <file> --root <root>`; there is no install that changes this (see the `retrieval` skill's "Author the media transcript yourself (the default media path)" section, and `skills/retrieval/references/office-image-transcripts.md` for the full authoring guide) |
| my mp4 (or other audio/video file) shows as a stub in `sidecar --list` (`reason: agent-orchestrated`, `warning: audio/video files were indexed as agent-orchestrated stubs`) | Expected — audio/video has **no** machine extractor, and an agent can't read it natively either (unlike docx/images); this is not an error state | Run an ASR tool via Bash (WhisperX, whisper.cpp, or `whisper`), then register the transcript: `retrieval sidecar --register <file> --transcript <file> --root <root>` (see the `retrieval` skill's "Audio and video: orchestrate an ASR tool, then register"). If no ASR tool is available, leave the stub as-is; step-by-step: [Transcribe audio and video](../how-to/transcribe-audio-video.md) |

In the `pi-serini` row above, both spellings are correct: `pi-serini` is
the retriever key, `pyserini` is the Python library/extra it needs.

For anything not covered here, the five knowledge skills
(`skills/lexical-retrieval-usage`, `skills/dense-retrieval-usage`,
`skills/lucene-retrieval-usage`, `skills/code-retrieval-usage`,
`skills/hybrid-retrieval-usage`) each have a "Graceful degradation" section
with the exact failure modes for that method.

## MCP server

| Symptom | Cause | Fix |
|---|---|---|
| stderr: `agentic-retrieval MCP server needs the 'mcp' extra: ...`, exit code 1 | `retrieval mcp` was run without the `mcp` extra synced | `uv sync --project engine --extra mcp` (or `--extra all`), or launch through `scripts/retrieval-mcp.sh`, which adds the extra |
| stderr: `agentic-retrieval MCP: 'uv' not found on PATH` | `scripts/retrieval-mcp.sh` can't find `uv` | Install `uv`, make it visible to Claude Code's environment, and restart |
| Search results come only from `lexical`; the other strategies are listed in `skipped` with `not indexed - call index_project(...)` | Those strategies' caches haven't been built | Call `index_project(retriever='all')` (slow on large repos, one-off) or run `retrieval index` from the CLI |
| `error_type: BuildInProgress` | Another `index_project` build is already running | Poll `index_status` and retry when it finishes. Searches in the meantime use the existing caches, report `build_in_progress: true` and skip `pi-serini`. A build can't be cancelled from MCP; stop the server process to abort it |
| `error_type: OptInRequired` | `lexical+ctx` (paid LLM contextualizer) is never built through MCP; `lexical`/`all` builds also refuse when the lexical cache slot holds a `lexical+ctx` index | Build it from the CLI after explicit opt-in: `retrieval index --retriever lexical+ctx` |
| Results report `stale` cache names and a hint to refresh | With the default `stale_ok=None`, only the plain `lexical` cache is refreshed; heavy caches are searched as-is | Call `index_project` to refresh, or pass `stale_ok=False` to search |
| The server indexes the wrong directory | Root resolution picked up a different directory | Check the `agentic-retrieval MCP: root=...` line on stderr; set `RETRIEVAL_ROOT` or pass `--root` (order: `--root`, `RETRIEVAL_ROOT`, `CLAUDE_PROJECT_DIR`, cwd) |

## Plugin install

| Symptom | Fix |
|---|---|
| Skills don't appear after installing | `rm -rf ~/.claude/plugins/cache`, restart Claude Code, then reinstall the plugin. |
| `/plugin` isn't a recognized command | Your Claude Code build predates plugin support — update Claude Code. |

## Next steps

- [Index and query how-to](../how-to/index-and-query.md)
- [Install the plugin](../how-to/install.md)
- [Environment variables](environment-variables.md)
- [MCP server](../how-to/integrate-coding-agents.md#mcp-server)
