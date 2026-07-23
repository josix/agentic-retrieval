# `retrieval.project_loader`

`discover_files` / `load_documents` / `load_chunks` / `load_chunk_documents` /
`load_ast_chunk_documents` over a project root. `load_chunk_documents` is the
production loader most retrievers index over — it returns one
chunk-granularity `Document` per span (`docid = "{path}:{start}-{end}"`).
`load_ast_chunk_documents` is the AST-boundary analog `TreeSitterRetriever`
indexes over: chunks at tree-sitter node boundaries when a file's language is
supported, carrying an enclosing function/class breadcrumb (`context`), and
falling back to the line-based chunker per file otherwise.

::: retrieval.project_loader

## Keyword-only overrides

| kwarg | Default | Purpose |
|---|---|---|
| `extensions` | `DEFAULT_EXTENSIONS` | Allowed file suffixes (e.g. `frozenset({'.md'})` to index only Markdown) |
| `exclude_dirs` | `DEFAULT_EXCLUDE_DIRS` | Directory names pruned during traversal (e.g. add `'docs-archive'`) |
| `exclude_globs` | `DEFAULT_EXCLUDE_GLOBS` | Filename deny-list globs (secret-looking names) |
| `include_basenames` | `DEFAULT_INCLUDE_BASENAMES` | Extensionless basenames allowed regardless of `extensions` (e.g. add `'justfile'`) |
| `max_bytes` | `MAX_FILE_BYTES` (1 MB) | Per-file size cap |

See [customize indexing](../../how-to/customize-indexing.md) for runnable
examples of each override.
