# `retrieval.record_loaders`

Registry of record-oriented loaders. A record loader maps a file suffix to a
callable `load(docid, text, url) -> list[Document]` that yields one
chunk-granularity `Document` per logical record (a gettext entry, a CSV row,
a subtitle cue). Stdlib-only.

`retrieval.project_loader` consults `lookup(path)` in both
`load_chunk_documents` and `load_ast_chunk_documents`, and the CLI prints
`detail_lines(metadata)` under each hit in text mode, so the engine core
contains no format-specific code. `retrieval.po_loader` registers `.po` this
way; it is listed in `BUILTIN_LOADER_MODULES` and imported lazily on first
lookup.

Each loader declares `detail_keys` (rendered as `label: value`) or a
`describe` hook for custom text-mode lines. JSON output is unaffected.

::: retrieval.record_loaders

## Next steps

- [Customize indexing: record loaders](../../how-to/customize-indexing.md#record-loaders-one-document-per-record)
- [Reference: po_loader API](po_loader.md)
- [Reference: project_loader API](project_loader.md)
