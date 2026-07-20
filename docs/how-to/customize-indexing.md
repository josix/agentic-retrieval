# Customize what gets indexed

`discover_files` / `load_documents` / `load_chunks`
(`retrieval/project_loader.py`) all accept the same keyword-only overrides:

| kwarg | Default | Purpose |
|---|---|---|
| `extensions` | `DEFAULT_EXTENSIONS` | Allowed file suffixes (e.g. `frozenset({'.md'})` to index only Markdown) |
| `exclude_dirs` | `DEFAULT_EXCLUDE_DIRS` | Directory names pruned during traversal (e.g. add `'docs-archive'`) |
| `exclude_globs` | `DEFAULT_EXCLUDE_GLOBS` | Filename deny-list globs (secret-looking names) |
| `include_basenames` | `DEFAULT_INCLUDE_BASENAMES` | Extensionless basenames allowed regardless of `extensions` (e.g. add `'justfile'`) |
| `max_bytes` | `MAX_FILE_BYTES` (1 MB) | Per-file size cap |

## Extend, don't replace

Extend a `DEFAULT_*` constant with `|` rather than replacing it, so you keep
the built-in exclusions:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJECT_ROOT="$PROJECT_ROOT" uv run --project engine --extra all python - <<'PY'
import os

from retrieval.project_loader import DEFAULT_EXCLUDE_DIRS, load_documents

root = os.environ["PROJECT_ROOT"]
no_archive = load_documents(root, exclude_dirs=DEFAULT_EXCLUDE_DIRS | {"docs-archive"})
print("excluding docs-archive:", len(no_archive))
PY
```

Replacing `DEFAULT_EXCLUDE_DIRS` outright (instead of unioning with `|`)
silently drops the built-in exclusions — `.git`, `.venv`, `node_modules`,
etc. would then be walked and indexed.

## Default exclusions summary

- VCS/venv/build/cache directories (`.git`, `.venv`, `node_modules`,
  `__pycache__`, `dist`, `build`, `.ruff_cache`, etc.)
- Filenames that look like secrets by name (`.env`, `*.pem`, `*.key`,
  `id_rsa*`, `*credentials*`, `*secret*`, etc. — filename matching only, not
  content scanning)
- Binary/undecodable files
- Anything over 1 MB

!!! warning
    The filename deny-list (`DEFAULT_EXCLUDE_GLOBS`) is a **best-effort**
    guard against accidentally indexing credential files by name — it is
    not content scanning. A file named `notes.txt` containing an API key
    will still be indexed; do not rely on this module for secret detection.

Extensionless well-known project files (`Dockerfile`, `Makefile`, `LICENSE`,
`README`, `CHANGELOG`, etc.) are included via a basename allowlist, since the
extension allowlist alone would otherwise skip them.

Full kwargs reference and more examples:
`skills/lexical-retrieval-usage/SKILL.md` ("Customizing file discovery").

## Next steps

- [Reference: project_loader API](../reference/api/project_loader.md)
- [Architecture](../concepts/architecture.md)
