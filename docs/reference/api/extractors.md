# `retrieval.extractors`

Sidecar-transcript extraction for non-text-native document formats (PDF).
Stdlib-only at import scope — this module must import cleanly with zero
optional extras installed, same guarantee as the rest of the default
pipeline; the `pypdf` backend is imported lazily, only inside the functions
that actually need it.

A PDF's extracted text is written once to a Markdown "sidecar" file under
`<project-root>/.agentic-retrieval/extracted/<rel-path>.md` (always under
the project root, deliberately ignoring `RETRIEVAL_INDEX_DIR` — a sidecar is
a citation target a coding agent `Read()`s by project-relative path, so it
must live in the tree being indexed even when the *cache* itself is
redirected elsewhere). A content-hash + extractor-version manifest
(`manifest.json`, alongside the sidecars) makes re-extraction a no-op on
unchanged files across the multiple loader passes `index --auto` performs
per run.

Every failure mode (encrypted, malformed, empty, no-text-layer,
backend-missing) still produces a non-empty, human-readable stub sidecar —
an empty sidecar would yield zero chunks and silently vanish from every
index.

::: retrieval.extractors

## Stub taxonomy

| `reason` | When |
|---|---|
| `encrypted` | Password-protected; the empty-password decrypt attempt failed |
| `crypto-unavailable` | Encrypted with a method needing the optional `cryptography` package |
| `empty` | Zero pages |
| `no-text-layer` | Scanned/image-only document (no OCR performed) |
| `malformed` | Corrupted or unsupported PDF structure |
| `backend-missing` | `pypdf` is not installed |

## Backend-missing degradation vs. hard failure

`index`/`query` never hard-fail on a missing `pypdf` backend — a PDF
without a usable backend still indexes as a searchable, backend-missing
stub, with a single `warning: pypdf is not installed` line printed to
stderr per process (`_warn_backend_missing`). The `retrieval extract`
subcommand is the one place a missing backend *does* hard-fail, via an
explicit preflight `require_extractors()` call — see [CLI reference:
`extract`](../cli.md#extract).

## Next steps

- [Customize indexing: PDF auto-indexing](../../how-to/customize-indexing.md#pdf-auto-indexing)
- [Persistence and cache: PDF sidecars](../persistence-and-cache.md)
- [Reference: project_loader API](project_loader.md)
