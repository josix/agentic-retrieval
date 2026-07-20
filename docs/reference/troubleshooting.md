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
