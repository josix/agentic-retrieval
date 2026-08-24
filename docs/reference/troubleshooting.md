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
| PDF search results are placeholder stubs (`warning: pypdf is not installed`) | `pdf` extra not synced — the expected first state under the skill's default sync, which delegates media extraction to the agent | Author the transcript yourself: `retrieval sidecar --register <pdf> --transcript <file> --root <root>` (no install required — see the `retrieval` skill's "Without the `pdf` extra: author the transcript yourself"). Alternatively opt in to machine extraction: `uv sync --project engine --extra pdf` and reindex |
| docx/pptx/xlsx/image search results are placeholder stubs (`warning: media files were indexed as agent-transcribable stubs`) | Expected — these formats have **no** machine extractor at all, `pdf` extra or not | Author the transcript yourself: `retrieval sidecar --register <file> --transcript <file> --root <root>`; there is no install that changes this (see the `retrieval` skill's "Without the `pdf` extra: author the transcript yourself") |
| my mp4 (or other audio/video file) shows as a stub in `sidecar --list` (`reason: agent-orchestrated`, `warning: audio/video files were indexed as agent-orchestrated stubs`) | Expected — audio/video has **no** machine extractor, and an agent can't read it natively either (unlike docx/images); this is not an error state | Run an ASR tool via Bash (WhisperX, whisper.cpp, or `whisper`), then register the transcript: `retrieval sidecar --register <file> --transcript <file> --root <root>` (see the `retrieval` skill's "Audio and video: orchestrate an ASR tool, then register"). If no ASR tool is available, leave the stub as-is |

In the `pi-serini` row above, both spellings are correct: `pi-serini` is
the retriever key, `pyserini` is the Python library/extra it needs.

For anything not covered here, the four knowledge skills
(`skills/lexical-retrieval-usage`, `skills/dense-retrieval-usage`,
`skills/lucene-retrieval-usage`, `skills/hybrid-retrieval-usage`) each have
a "Graceful degradation" section with the exact failure modes for that
method.

## Plugin install

| Symptom | Fix |
|---|---|
| Skills don't appear after installing | `rm -rf ~/.claude/plugins/cache`, restart Claude Code, then reinstall the plugin. |
| `/plugin` isn't a recognized command | Your Claude Code build predates plugin support — update Claude Code. |

## Next steps

- [Index and query how-to](../how-to/index-and-query.md)
- [Install the plugin](../how-to/install.md)
- [Environment variables](environment-variables.md)
