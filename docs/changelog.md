# Changelog

Entries are tagged by component: **\[engine\]** for the Python package under
`engine/`, **\[plugin\]** for Claude Code plugin primitives (skills, agents,
commands, manifests), **\[docs\]** for documentation-only changes. The plugin
and the engine are versioned in lockstep — a single version number covers
both.

## Unreleased

- **\[engine\]** Pluggable tokenizer modes (`plain`/`code`) on `TfidfIndex` and
  `BM25Index`. `"code"` mode emits each whole token's lowercased subtokens
  (split on `_`/`-`/`.` and camelCase/PascalCase boundaries) after the whole
  token, so identifier-heavy corpora can match on subtoken vocabulary (e.g.
  `getUserById` → `user`/`by`/`id`) without losing exact-identifier matches.
  `"plain"` stays byte-identical to the prior tokenizer. The mode is
  persisted in `to_dict`/`from_dict` and never re-derived at query time.
- **\[engine\] Breaking: cache schema bump (v2 -> v3) for `LexicalRetriever`
  and `HybridRetriever`.** Tier-A hyperparameters (`bm25_k1`/`bm25_b`/
  `tokenizer`/`model_name`/`bit_width`/`lucene_k1`/`lucene_b` and a
  chunk-size policy) are now threaded from CLI flags (or auto-derived corpus
  signals) through to their consumers and recorded in each cache's `meta`,
  so query time never re-derives them; a v2 on-disk cache for either
  retriever is treated as unrecognized and auto-rebuilt on the next
  `index`/`query` — no manual cache-clearing needed. New `build_retriever`/
  `resolve_ctor_kwargs` map a filtered `params` dict onto each retriever's
  declared `ACCEPTS` set. New `retrieval.autotune` module (stdlib-only)
  collects corpus signals and resolves the full hyperparameter dict via
  `resolve_params()`, with a static-default fallback on an empty corpus. New
  `retrieval tune` subcommand reports auto-derived params without writing
  anything; `index`/`query` gain `--auto` plus per-hyperparameter flags
  (`--bm25-k1`, `--tokenizer`, `--bit-width`, etc.), `query`'s `--top-k`
  default becomes `None` (explicit > auto-resolved > `5`), `eval` gains
  `--auto` and records `auto_params`, and `stats` echoes `hyperparams`/
  `corpus_stats` when present. Hyperparameters recorded this way are now
  preserved across both query-triggered and index-triggered cache rebuilds
  (a bare `query` or bare `index`/`index --force` with no explicit flags no
  longer silently reverts a cache's recorded `bm25_k1`, `tokenizer`, etc. to
  static defaults) — an explicit flag value still takes precedence and
  overwrites the recovered value, and a from-scratch index with no prior
  `meta` is unaffected.
- **\[engine\] \[docs\]** Hyperparameter decisions delegated to the invoking
  agent. `resolve_params` now also decides the turbovec/hybrid embedding
  model from corpus content and size (`_decide_embed_model`):
  code-dominated corpora (>0.6 code bytes) get a code-search-trained
  embedder, small corpora (≤2000 chunks) get the higher-quality `mpnet`,
  large corpora get the fast `MiniLM`, and an empty corpus degrades to the
  `MiniLM` static default; `bit_width` is now sized from the chosen (or
  overridden) model's embedding dimensionality instead of a hardcoded 384.
  `skills/retrieval/SKILL.md` Step 2 and the command dispatcher
  (`commands/retrieval.md`) now instruct the agent to decide hyperparameters
  inline at index/query time — `index` with `--auto` as the corpus-
  calibrated baseline plus per-flag explicit overrides, `query` with
  `--auto`, `top-k`, `--weights`, and `--pool` — including a decision table
  mapping corpus/query observations to concrete flags and the persistence
  semantics (decide once, survives rebuilds, changed values trigger a
  selective rebuild).
- **\[engine\]** `retrieval.__version__` realigned with `pyproject.toml`
  (both `0.6.0`); a new regression test parses `pyproject.toml`'s version
  field by regex and asserts it matches `retrieval.__version__` so this
  can't silently drift again. This explains why on-disk caches built during
  the 0.6.0 development window may show differing `engine_version` values
  in their `meta.json` depending on exactly which commit produced them.

## 0.6.0 — 2026-07-25

- **\[plugin\] New default deep-answer workflow (Q/R/T/C/S).** The `retrieval`
  skill's Step 3 now runs a phased Decompose → Retrieve-per-sub-question →
  Trace → Coverage-gate → Synthesize loop by default for explanatory/
  comprehensiveness questions (with a shallow triage fast-path for bare
  locate-a-symbol asks), replacing the previous four-step loop with an
  explicit coverage checklist that must be satisfied before answering.
  **New optional `retrieval-tracer` agent** (`agents/retrieval-tracer.md`)
  for fanning out Phase T across sub-aspects on broad questions (≥4
  independent sub-aspects), returning detailed `file:line` traces to be
  stitched together by the caller. `docs/how-to/consolidated-query.md` and
  `docs/how-to/integrate-coding-agents.md` document the workflow's use of
  the `--output` JSON envelope as the per-sub-question seed carrier,
  including a non-Claude-agent variant using only the CLI. **No CLI
  output, JSON envelope, or engine logic changed** — this is entirely
  plugin-primitive (skill/agent/doc) guidance. `--queries-file`/
  `--path-prefix` engine flags were considered to support this workflow
  natively but deferred; the workflow is achievable today with
  per-sub-question `query --output` calls.
- **\[docs\] Docs/guidance-only clarification.** The `retrieval` skill, the
  five per-method usage skills, and the `retrieval.consolidation`
  docstrings/docs now spell out that the consolidated list is scaffolding
  for exploration (not a finished answer to hand the user directly), and
  that `score`/`agreement`/`confidence` measure cross-retriever agreement on
  query-text match — not that a span is current, canonical, or
  non-deprecated. No CLI output, JSON envelope, or engine logic changed.

## 0.5.0 — 2026-07-24

- **\[engine\] Breaking:** the default persisted index cache moved from
  `~/.cache/agentic-retrieval/indexes/<project-key>` to
  `<project-root>/.agentic-retrieval`, resolved from the indexed root's
  absolute path so it follows the project regardless of where it's checked
  out. Existing caches under the old home-directory location are abandoned
  (not migrated) and lazily rebuilt at the new location on the next
  `index`/`query`. The `RETRIEVAL_INDEX_DIR` environment variable now
  instead selects a shared base directory (keyed by `project_key(root)`)
  rather than being the default location itself; its shared-base +
  project-key layout is otherwise unchanged. The new `.agentic-retrieval`
  cache dirname is excluded from file discovery by default (preventing a
  self-indexing feedback loop) and added to `.gitignore` so index blobs are
  never committed.

## 0.4.0 — 2026-07-24

- **\[engine\] Breaking:** the distribution package was renamed from
  `rag-retrieval` to `agentic-retrieval` (the `rag-retrieval` name was
  already taken on PyPI). The importable module (`retrieval`) and the
  `retrieval` console script are unchanged; only the distribution name, its
  self-referencing `all` extra, the release workflow's PyPI URL, and install
  instructions moved to `agentic-retrieval`. Any install command referencing
  the old `rag-retrieval` distribution name (e.g. `uv pip install
  rag-retrieval`, `uvx --from rag-retrieval`, or a `rag-retrieval[all] @
  git+...` requirement) must be updated to `agentic-retrieval`.
- **\[engine\] \[docs\]** New `retrieval eval` subcommand runs a labeled query
  set against every retriever plus consolidated fusion, reporting
  recall@k, nDCG@k, confidence-bucket precision (agreement-signal
  validity), and cold/warm search latency; entirely in-memory (no on-disk
  cache writes). Missing extras are skipped gracefully, mirroring the
  consolidated query path. Ships an in-repo fixture corpus with 10
  hand-labeled queries and extras-free tests, plus a new how-to guide
  (`docs/how-to/evaluate-retrievers.md`). `skills/retrieval/SKILL.md` now
  states what single-retriever mode trades away (the cross-retriever
  agreement/confidence signal) so agents can choose the opt-out
  deliberately; the consolidated default is unchanged.

## 0.3.0

- **\[engine\] Breaking: `retrieval query` default is now consolidated (all-retriever)
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
- **\[engine\] Weighted Reciprocal Rank Fusion.** `retrieval.fusion.reciprocal_rank_fusion`
  gained a trailing `weights: Optional[List[float]] = None` parameter;
  `None` (the default) weights every ranking `1.0`, numerically identical to
  the previous unweighted behavior, so every existing caller is unaffected.
- **\[engine\] New `retrieval.consolidation` module.** `ConsolidatedHit` dataclass +
  `consolidate(per_retriever_hits, *, k=60, weights=None,
  merge_adjacent=True) -> List[ConsolidatedHit]`: groups same-file
  overlapping/adjacent `SearchHit` spans across retrievers into one
  candidate, fuses each group's per-retriever ranks with weighted RRF, and
  returns a list sorted best-first, each hit carrying `provenance` (which
  retrievers found it), `agreement` (how many), and a `confidence` label
  (`high` if `agreement >= 2`; else `medium` if the sole contributing arm is
  a dense/Lucene arm — `turbovec`/`pi-serini`; else `low`).
- **\[engine\] Tree-sitter (AST-boundary chunking) retrieval strategy.** New
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
- **\[engine\] `site/` excluded from indexing by default.** `site` (MkDocs/static-site
  build output) joined `DEFAULT_EXCLUDE_DIRS` in `retrieval.project_loader`,
  so generated site assets no longer pollute the index. On-disk caches built
  before this change will read as stale on the next `query` (the content
  fingerprint now differs) and auto-rebuild — no manual cache-clearing.
  Override with the `exclude_dirs=` keyword if your project keeps real source
  under `site/`.
- **\[plugin\] Skill guidance: post-retrieval enrichment loop.** The `retrieval` skill
  gained a Step 3 framing retrieved spans as exploration seeds (read the span,
  follow references outward, re-query with the vocabulary a hit reveals, stop
  when the question is answered) plus a noisy/low-relevance recovery
  subsection. The four `*-retrieval-usage` skills and the integrate-coding-
  agents how-to mirror the same "seed, not final answer" framing; the earlier
  "no manual grepping needed" / "no re-grepping" stop-signal phrasing was
  removed.
- **\[engine\] Chunk-level indexing with file:line spans.** All five production
  retrievers now index chunk-granularity `Document`s
  (`retrieval.project_loader.load_chunk_documents`) instead of whole
  files: `docid` is `"{path}:{start}-{end}"`, and each `Document` carries
  `source_path`/`start_line`/`end_line` span metadata. Every retriever
  gained `search_detailed(query, top_k) -> List[SearchHit]`, resolving
  each result's span from that metadata (never by parsing the docid
  string). `search()` is now a thin `[h.docid for h in
  search_detailed(...)]` projection, kept for backward compatibility.
- **\[engine\] Chunker line-span tracking.** `retrieval.chunker.Chunk` gained
  `start_line`/`end_line` (1-based, inclusive); `chunk_document` tracks
  source line numbers for every paragraph and extends a chunk's span to
  include the heading line that introduced it.
- **\[engine\] Breaking: `retrieval query --json` output shape.** `results` was a
  flat list of docid strings; it is now a list of objects
  (`{"docid", "path", "start_line", "end_line", "rank"}`). Plain-text
  `query` output changed from one docid per line to one
  `path:start_line-end_line` span per line.
- **\[engine\] Cache schema bump (v1 -> v2).** Every retriever's `SCHEMA_VERSION`
  bumped to persist the new chunk-span `units`; a v1 on-disk cache is
  treated as unrecognized and auto-rebuilt on the next `index`/`query` —
  no manual cache-clearing needed.
- **\[engine\]** `retrieval index` prints `<name>: indexed <N> chunks` (was `... docs`);
  `retrieval stats` prints both `chunks:` (chunk count) and `files:`
  (distinct source-file count) instead of a single `docs:` line. The
  cache's `meta.json` gained a `file_count` field alongside the existing
  (now chunk-counting) `doc_count`.

## 0.2.0

- **\[engine\]** Persistent, on-disk index cache (`retrieval.persistence`) keyed by
  project path, with fingerprint-based staleness detection.
- **\[engine\]** New `retrieval` console-script CLI (`index` / `query` / `stats`),
  including `--force`, `--stale-ok`, and `--json`.
- **\[engine\]** `uv`-native setup — the engine's environment is managed entirely by
  `uv sync`/`uv run`; the deprecated no-`uv` fallback (`scripts/setup_venv.py`)
  has been removed.
- **\[engine\]** `all` extra — one-shot install of every optional retrieval extra
  (`local`, `remote`, `turbovec`, `pyserini`).

## 0.1.0

**\[engine\]** Initial vendored engine: `LexicalRetriever` (TF-IDF + BM25 + RRF),
`TurbovecRetriever` (dense ANN), `PiSeriniRetriever` (Lucene BM25), and the
heuristic/LLM contextualizers, with an in-memory-only engine API.

See [GitHub Releases](https://github.com/josix/agentic-retrieval/releases)
for the full release history and downloadable build artifacts.
