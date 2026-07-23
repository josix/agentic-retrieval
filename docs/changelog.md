# Changelog

## Unreleased

- **Breaking: `retrieval query` default is now consolidated (all-retriever)
  mode.** With no `--retriever` flag (or the new explicit `--retriever all`
  alias), `query` loads every available strategy in `_DEFAULT_INDEX_SET`,
  searches each, and merges/fuses them into a single deduplicated, ranked,
  explainable list via the new `retrieval.consolidation.consolidate` —
  same-file overlapping/adjacent spans across retrievers merge into one
  candidate, and each result carries `score`/`provenance`/`agreement`/
  `confidence`/`contributors`. A missing backend is skipped (never a hard
  failure) as long as `lexical` consolidates successfully. New consolidated-
  mode-only flags: `--weights "name:w,..."` (per-retriever RRF weight
  override) and `--output PATH` (also persist the JSON envelope to disk).
  **`--retriever <name>` (single strategy) is unaffected** — its plain-text
  and `--json` output stay byte-identical to before this change. See
  [Consolidated query](how-to/consolidated-query.md).
- **Weighted Reciprocal Rank Fusion.** `retrieval.fusion.reciprocal_rank_fusion`
  gained a trailing `weights: Optional[List[float]] = None` parameter;
  `None` (the default) weights every ranking `1.0`, numerically identical to
  the previous unweighted behavior, so every existing caller is unaffected.
- **New `retrieval.consolidation` module.** `ConsolidatedHit` dataclass +
  `consolidate(per_retriever_hits, *, k=60, weights=None,
  merge_adjacent=True) -> List[ConsolidatedHit]`: groups same-file
  overlapping/adjacent `SearchHit` spans across retrievers into one
  candidate, fuses each group's per-retriever ranks with weighted RRF, and
  returns a list sorted best-first, each hit carrying `provenance` (which
  retrievers found it), `agreement` (how many), and a `confidence` label
  (`high` if `agreement >= 2`; else `medium` if the sole contributing arm is
  a dense/Lucene arm — `turbovec`/`pi-serini`; else `low`).
- **Tree-sitter (AST-boundary chunking) retrieval strategy.** New
  `retrieval.ast_chunker` module implements cAST-style (arXiv 2506.15655)
  split-then-merge chunking at AST node boundaries, via
  `tree-sitter-language-pack` (new `treesitter` extra). Every chunk carries a
  dotted `context` breadcrumb (enclosing function/class path, e.g.
  `"Bar.baz"`). `retrieval.project_loader.load_ast_chunk_documents` is the
  AST-boundary analog of `load_chunk_documents`, falling back to the
  line-based chunker per file when a suffix's language is unmapped or a
  file's AST chunking comes back empty. New `TreeSitterRetriever`
  (REGISTRY key `treesitter`) subclasses `LexicalRetriever`, ranking with the
  same TF-IDF + BM25 + RRF over AST chunks with the breadcrumb prefixed into
  the ranked text — the retriever class itself needs no optional
  dependency, only the loader does. `Document` and `SearchHit` gained an
  optional, trailing-default `context: str = ""` field.
- **`site/` excluded from indexing by default.** `site` (MkDocs/static-site
  build output) joined `DEFAULT_EXCLUDE_DIRS` in `retrieval.project_loader`,
  so generated site assets no longer pollute the index. On-disk caches built
  before this change will read as stale on the next `query` (the content
  fingerprint now differs) and auto-rebuild — no manual cache-clearing.
  Override with the `exclude_dirs=` keyword if your project keeps real source
  under `site/`.
- **Skill guidance: post-retrieval enrichment loop.** The `retrieval` skill
  gained a Step 3 framing retrieved spans as exploration seeds (read the span,
  follow references outward, re-query with the vocabulary a hit reveals, stop
  when the question is answered) plus a noisy/low-relevance recovery
  subsection. The four `*-retrieval-usage` skills and the integrate-coding-
  agents how-to mirror the same "seed, not final answer" framing; the earlier
  "no manual grepping needed" / "no re-grepping" stop-signal phrasing was
  removed.
- **Chunk-level indexing with file:line spans.** All five production
  retrievers now index chunk-granularity `Document`s
  (`retrieval.project_loader.load_chunk_documents`) instead of whole
  files: `docid` is `"{path}:{start}-{end}"`, and each `Document` carries
  `source_path`/`start_line`/`end_line` span metadata. Every retriever
  gained `search_detailed(query, top_k) -> List[SearchHit]`, resolving
  each result's span from that metadata (never by parsing the docid
  string). `search()` is now a thin `[h.docid for h in
  search_detailed(...)]` projection, kept for backward compatibility.
- **Chunker line-span tracking.** `retrieval.chunker.Chunk` gained
  `start_line`/`end_line` (1-based, inclusive); `chunk_document` tracks
  source line numbers for every paragraph and extends a chunk's span to
  include the heading line that introduced it.
- **Breaking: `retrieval query --json` output shape.** `results` was a
  flat list of docid strings; it is now a list of objects
  (`{"docid", "path", "start_line", "end_line", "rank"}`). Plain-text
  `query` output changed from one docid per line to one
  `path:start_line-end_line` span per line.
- **Cache schema bump (v1 -> v2).** Every retriever's `SCHEMA_VERSION`
  bumped to persist the new chunk-span `units`; a v1 on-disk cache is
  treated as unrecognized and auto-rebuilt on the next `index`/`query` —
  no manual cache-clearing needed.
- `retrieval index` prints `<name>: indexed <N> chunks` (was `... docs`);
  `retrieval stats` prints both `chunks:` (chunk count) and `files:`
  (distinct source-file count) instead of a single `docs:` line. The
  cache's `meta.json` gained a `file_count` field alongside the existing
  (now chunk-counting) `doc_count`.

## 0.2.0

- Persistent, on-disk index cache (`retrieval.persistence`) keyed by
  project path, with fingerprint-based staleness detection.
- New `retrieval` console-script CLI (`index` / `query` / `stats`),
  including `--force`, `--stale-ok`, and `--json`.
- `uv`-native setup — the engine's environment is managed entirely by
  `uv sync`/`uv run`; the deprecated no-`uv` fallback (`scripts/setup_venv.py`)
  has been removed.
- `all` extra — one-shot install of every optional retrieval extra
  (`local`, `remote`, `turbovec`, `pyserini`).

## 0.1.0

Initial vendored engine: `LexicalRetriever` (TF-IDF + BM25 + RRF),
`TurbovecRetriever` (dense ANN), `PiSeriniRetriever` (Lucene BM25), and the
heuristic/LLM contextualizers, with an in-memory-only engine API.

See [GitHub Releases](https://github.com/josix/agentic-retrieval/releases)
for the full release history and downloadable build artifacts.
