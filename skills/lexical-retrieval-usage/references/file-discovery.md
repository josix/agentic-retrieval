# Customizing file discovery

Referenced from `../SKILL.md` ("Customizing file discovery").

`discover_files` / `load_documents` / `load_chunks`
(`retrieval/project_loader.py`) all accept the same keyword-only overrides
(forwarded via `**kw`): `extensions`, `exclude_dirs`, `exclude_globs`,
`include_basenames`, `max_bytes`. Import the matching `DEFAULT_*` constant and
extend it with `|` rather than replacing it outright, so you keep the
built-in secret/VCS/build exclusions:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_INCLUDE_BASENAMES,
    load_documents,
)

root = os.environ["PROJECT_ROOT"]

# Only index Markdown files.
md_only = load_documents(root, extensions=frozenset({".md"}))
print("md-only docs:", len(md_only))

# Extend (not replace) the default excluded directories.
no_archive = load_documents(
    root, exclude_dirs=DEFAULT_EXCLUDE_DIRS | {"docs-archive"}
)
print("excluding docs-archive:", len(no_archive))

# Extend the extensionless-basename allowlist (e.g. a `justfile`).
with_justfile = load_documents(
    root, include_basenames=DEFAULT_INCLUDE_BASENAMES | {"justfile"}
)
print("including justfile:", len(with_justfile))

# Raise the per-file size cap above the 1 MB default.
larger_files = load_documents(root, max_bytes=5_000_000)
print("max_bytes=5MB docs:", len(larger_files))
PY
```

`discover_files` takes the same kwargs (it returns `Path` objects instead of
`Document`s) and `load_chunks` forwards them straight through to
`load_documents` — see `chunk-level-indexing.md` for a `load_chunks` example.
