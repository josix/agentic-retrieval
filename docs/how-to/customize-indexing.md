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
| `extract_max_bytes` | `extractors.EXTRACT_MAX_BYTES` (25 MB) | Per-file size cap for extractable suffixes (PDF + agent-only media) — see [PDF auto-indexing](#pdf-auto-indexing) below |

## PDF auto-indexing

PDFs — and a second tier of agent-only media (`.docx`, `.pptx`, `.xlsx`,
`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) — under a project root are indexed
automatically: `extractors.EXTRACTABLE_EXTENSIONS` (all of the above) is
part of `DEFAULT_EXTENSIONS`, with no flag needed. Each such file is routed
through `retrieval.extractors` for a cached, sidecar-transcript `Document`
*before* the usual text-decode step, so it's chunked and searched as prose
rather than being silently skipped as binary.

- **Two size caps.** A PDF or media file is capped at `extract_max_bytes`
  (25 MB) rather than the 1 MB `max_bytes` plain-text cap — these files are
  typically much larger than source files, and applying the small default
  cap would silently exclude every real-world one.
- **Sidecar location and citation contract.** Each file's extracted text is
  written once to `<project-root>/.agentic-retrieval/extracted/<rel-path>.md`
  — always under the project root, deliberately ignoring
  `RETRIEVAL_INDEX_DIR`, since a sidecar is a citation target a coding agent
  `Read()`s by project-relative path. Search hits over such a file therefore
  cite the sidecar's path (e.g. `docs/paper.pdf` indexes as
  `.agentic-retrieval/extracted/docs/paper.pdf.md`), not the original file.
  A content-hash + extractor-version manifest alongside the sidecars makes
  re-extraction a no-op on unchanged files across `index --auto`'s multiple
  loader passes.
- **`--no-pdf`.** Pass `--no-pdf` to `retrieval index` to exclude PDFs
  *and* every other sidecar-routed media suffix from discovery entirely (an
  escape hatch for the auto-activated pipeline). The choice is persisted in
  the cache's meta and therefore sticky — a later flag-less `index`/`query`
  call keeps honoring it without needing the flag again.
- **Backend-missing stubs (PDF only).** Without the `pdf` extra (`pypdf`,
  included in the `all` extra) installed, every PDF still indexes — as a
  searchable placeholder stub naming the source file, instead of a real
  transcript — and a single `warning: pypdf is not installed` line prints
  to stderr per process. `index`/`query` never hard-fail on a missing
  `pypdf` backend; only the `retrieval extract` subcommand does (see
  [Reference: CLI](../reference/cli.md)).
- **Agent-only stubs (no machine extractor at all).** `.docx`, `.pptx`,
  `.xlsx`, and the image suffixes have no machine extractor — installing
  `pypdf` has no effect on them. Every such file always indexes as an
  `"agent-only"`-reason stub, and a single `warning: media files were
  indexed as agent-transcribable stubs` line prints to stderr per process.
  `retrieval sidecar --register` is the only way to index their real
  content.
- **Agent-authored transcripts (no-install alternative for PDF; the only
  path for agent-only media).** Instead of installing `pypdf`, an agent can
  read a PDF (or the original docx/pptx/xlsx/image) natively and
  hand-author its transcript, then register it directly: `retrieval
  sidecar --register <file> --transcript <file> --root <root>`. This
  writes the same kind of sidecar as `ensure_sidecar` would, stamped
  `extractor: agent-authored/1` instead of `pypdf-text/1`, and is a
  **durable** sidecar — it is not silently replaced if `pypdf` later gets
  installed (and never for agent-only media, which has no pypdf equivalent
  to fall back to); only an explicit `retrieval extract --force` overwrites
  a PDF's (with a warning; `extract` never touches agent-only media at
  all). It also mixes into the corpus fingerprint on its own (via the
  transcript's content hash), so re-registering a changed transcript over
  an otherwise-unchanged source file still triggers a reindex. See
  [Reference: CLI](../reference/cli.md#sidecar) and the `retrieval` skill's
  "Without the `pdf` extra: author the transcript yourself" section for the
  full workflow.

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
- Anything over 1 MB — except an extractable suffix (PDF + agent-only
  media), capped at 25 MB instead (see [PDF auto-indexing](#pdf-auto-indexing)
  above)

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
