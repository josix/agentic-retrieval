# Changelog

Entries are tagged by component: **\[engine\]** for the Python package under
`engine/`, **\[plugin\]** for Claude Code plugin primitives (skills, agents,
commands, manifests), **\[docs\]** for documentation-only changes. The plugin
and the engine are versioned in lockstep — a single version number covers
both.

## 0.11.0 — 2026-08-24

- **\[engine\] Caption files (`.srt`/`.vtt`) as tier-1 machine-extracted
  media** — a new stdlib-only extractor (`captions/1`,
  `CAPTIONS_EXTRACTOR_VERSION`) converts SRT/VTT cues into the sidecar
  transcript format: adjacent cues merge into `[HH:MM:SS]`-prefixed
  paragraphs, and a new `## [HH:MM:SS] <label>` section heading opens
  roughly every 180s. No third-party subtitle library, no new extra.
  `.srt`/`.vtt` are now part of `MACHINE_EXTRACTABLE_EXTENSIONS` and
  `DEFAULT_EXTENSIONS` — discovered and indexed automatically, same as a
  PDF, with `retrieval extract` pre-warming them too.
- **\[engine\] Third media tier: agent-orchestrated audio/video** — new
  `AGENT_ORCHESTRATED_EXTENSIONS` (`.mp4`, `.mov`, `.mkv`, `.webm`, `.mp3`,
  `.m4a`, `.wav`, `.flac`), folded into `EXTRACTABLE_EXTENSIONS`. Unlike the
  agent-only tier (docx/pptx/xlsx/images, which an agent reads/views
  natively), an agent cannot read a video/audio file's bytes at all — it
  must orchestrate an external ASR tool via Bash (WhisperX, whisper.cpp, or
  `whisper`) and register the result via `retrieval sidecar --register`,
  same mechanism as every other agent-authored sidecar. Every such file
  indexes as an `"agent-orchestrated"`-reason stub until registered; see
  the `retrieval` skill's new "Audio and video: orchestrate an ASR tool,
  then register" section for the full workflow, including anti-fabrication
  guidance specific to ASR output and the caption/video double-counting
  pitfall.
- **\[engine\] Stat-only source identity for tier 3** — `ensure_sidecar`,
  `register_sidecar`, and `sidecar --list` never read an audio/video
  source's bytes: a new shared `_source_identity` helper computes
  `sha256("stat/1|{size}|{mtime_ns}")` from a single `os.stat()` call
  instead for `AGENT_ORCHESTRATED_EXTENSIONS`, stored under the existing
  `sha256` manifest key with a new sibling `identity` field (`"stat/1"` for
  tier 3, `"sha256/1"` — a real content hash, unchanged behavior — for
  every other tier). Audio/video is also exempted from any file-size cap at
  discovery (`project_loader._is_eligible_file`), since the engine never
  allocates a buffer to hash bytes it would then discard. Trade-off: an
  in-place edit preserving both size and mtime produces a false cache hit
  under `"stat/1"` (impossible under `"sha256/1"`); `retrieval extract
  --force` is the escape hatch.
- **\[engine\] Generalized unit-heading regex** — `_PAGE_HEADING_RE` is now
  `_UNIT_HEADING_RE`, matching both `## Page N` (PDF) and `## [HH:MM:SS]
  label` (caption/media) headings; `_truncate_at_page_boundary` is now
  `_truncate_at_unit_boundary` and truncates a time-coded transcript at a
  whole-section boundary, never mid-paragraph.
- **\[engine\] `require_extractors` no longer conflates "has a machine
  extractor" with "needs pypdf"** — it now checks the new
  `PYPDF_EXTENSIONS` (`{".pdf"}`) instead of the broader
  `MACHINE_EXTRACTABLE_EXTENSIONS` (which now also includes captions), so a
  caption-only tree's `retrieval extract` succeeds without `pypdf`
  installed. `register_extractor` gained an optional `version=` kwarg
  (backing a new `_EXTRACTOR_VERSIONS` per-suffix map and
  `extractor_version_for(path)`) so a non-pypdf extractor stamps its own
  manifest `extractor_version` without bumping every other suffix's cache.
- **\[engine\] Uncapped tier-3 discovery** — `discover_files` no longer
  applies any size cap to `AGENT_ORCHESTRATED_EXTENSIONS` files (see stat-only
  identity above); a 40MB `.mp4` is now discovered where a 30MB `.pdf`
  still is not.

**Upgrade note:** a corpus containing `.srt`/`.vtt`, or any
`AGENT_ORCHESTRATED_EXTENSIONS` audio/video file, under its indexed root
reindexes once, automatically, on the next `index`/`query` after upgrading
— these suffixes are newly discovered and fingerprinted. A corpus with none
of these file types fingerprints byte-identically to the pre-0.11.0 format.
This is the same one-time fingerprint churn 0.9.0 introduced when
agent-only media suffixes were added to discovery.

## 0.10.0 — 2026-08-21

- **\[engine\] Terraform/HCL indexing** — `.tf`, `.tfvars`, and `.hcl` are
  now in `DEFAULT_EXTENSIONS`, so Terraform and HCL files pass file
  discovery; they are also added to `chunker.CONFIG_SUFFIXES`, so they
  chunk under the structured-config policy (`config_chars`) like
  YAML/TOML. Re-index existing projects to pick them up.

## 0.9.0 — 2026-08-17

- **\[engine\] Agent-authored PDF sidecar transcripts** — on clients without
  the `pdf` extra, the invoking coding agent can now build a usable PDF
  index itself instead of leaving placeholder stubs. New
  `retrieval.extractors.register_sidecar(root, source, transcript)` writes
  a durable sidecar from a hand-authored transcript (no `pypdf` import on
  any path), stamped `extractor_version: "agent-authored/1"`
  (`AGENT_EXTRACTOR_VERSION`) plus `authored_by: "agent"`, `authored_at`,
  and `sidecar_sha256` manifest fields. New `agent_sidecar_revision(entry)`
  and `sidecar_states(root, sources)` helpers back the CLI surface below.
  New `invalidate_process_cache(source)` drops a source's process-level
  memo so a freshly registered sidecar is served immediately instead of a
  stale in-process stub.
- **\[engine\]** New `retrieval sidecar` subcommand — `--list` reports
  every discovered PDF's sidecar state (`missing`/`outdated`/
  `agent-authored`/`stub`/`ok`); `--register SOURCE --transcript PATH|-`
  registers a hand-authored transcript (reading from a file or stdin).
  Unlike `extract`, `sidecar` never imports/requires `pypdf` on either
  mode.
- **\[engine\]** `retrieval.persistence.compute_fingerprint` now mixes an
  agent-authored PDF's `sidecar_sha256` into that file's fingerprint line
  (`"relpath|size|mtime|sidecar_sha256"` instead of the base
  `"relpath|size|mtime"`), so re-registering a changed transcript over an
  otherwise-unchanged source PDF still triggers a reindex. A corpus with no
  agent-authored entries fingerprints byte-identically to the pre-0.9.0
  format (invariant covered by a dedicated regression test).
- **\[engine\] Supersede policy:** an agent-authored sidecar is never
  silently replaced once `pypdf` becomes available —
  `needs_reextraction`/`_is_cache_hit`'s backend-missing check stay pinned
  on the literal `reason == "backend-missing"` string, which an
  agent-authored entry never has. The one escape hatch is `retrieval
  extract --force`, which now prints `warning: overwriting N
  agent-authored sidecar(s) with pypdf output` to stderr before doing so.
- **\[engine\]** `_is_cache_hit` hardened: `extractor_version` is now
  membership in `_ACCEPTED_EXTRACTOR_VERSIONS` (both pypdf and
  agent-authored) rather than equality against one constant, and a
  manifest entry missing `status`/`sidecar` is now treated as a cache miss
  (triggering re-extraction) instead of a latent `KeyError`.
- **\[engine\]** `require_extractors`'s guidance `RuntimeError` and the
  `backend-missing` stub's own text both now point at `retrieval sidecar
  --register` as the no-install alternative to the `pdf` extra.
- **\[plugin\]** `skills/retrieval/SKILL.md`: new "Without the `pdf`
  extra: author the transcript yourself" subsection walks the agent
  through detect -> read -> write -> register -> reindex -> verify, plus
  two new guardrails (never fabricate transcript content; never hand-edit
  a sidecar `.md` file).
- **\[plugin\] Media extraction is delegated to the agent by default** —
  the skill's Step 1 sync (and the `setup` dispatcher) now installs every
  strategy extra EXCEPT `pdf`, making agent-authored transcripts the
  standard media-indexing path rather than a fallback: after `index`, the
  `pypdf is not installed` warning is the agent's cue to run the sidecar
  workflow immediately, so the media index is built before any query
  needs it. Adding `--extra pdf` (or `setup pdf`/`setup all`) remains the
  explicit opt-in for pypdf machine extraction (bulk text-native corpora,
  headless `retrieval extract` pre-warms) — the engine's pypdf path is
  unchanged. Step 2's `uv run` snippets carry the same extras list as
  Step 1, since `uv run` re-syncs the environment to the extras named on
  each invocation.
- **\[docs\]** New `## sidecar` section in the CLI reference; agent-authored
  manifest keys and fingerprint-revision behavior documented in
  [persistence-and-cache.md](reference/persistence-and-cache.md);
  `register_sidecar`/`sidecar_states`/supersede policy documented in
  [api/extractors.md](reference/api/extractors.md); new troubleshooting
  entry for placeholder-stub PDF results; `commands/retrieval.md` and
  [customize-indexing.md](how-to/customize-indexing.md) point at the new
  workflow.
- **\[engine\] Sidecar mechanism extended to agent-only media** — the
  sidecar-extraction pipeline now also covers `.docx`, `.pptx`, `.xlsx`,
  `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` (`AGENT_ONLY_EXTENSIONS`), none
  of which have a machine extractor at all. Every such file always indexes
  as an `"agent-only"`-reason stub (distinct from PDF's `backend-missing`,
  so it never falsely self-heals once pypdf becomes available); a
  one-time-per-process stderr note points at `retrieval sidecar
  --register` as the only indexing path. `EXTRACTABLE_EXTENSIONS` is now
  the union of the new `MACHINE_EXTRACTABLE_EXTENSIONS` (`.pdf`) and
  `AGENT_ONLY_EXTENSIONS`, so discovery/fingerprinting/`register_sidecar`
  validation extend to the new suffixes automatically; `require_extractors`
  now only preflights `MACHINE_EXTRACTABLE_EXTENSIONS`. `retrieval extract`
  stays PDF-only — it never attempts agent-only media — and its `--prune`
  keep-set was fixed to cover every discovered sidecar-eligible file (not
  just PDFs), so a registered agent-only sidecar is no longer deleted by
  `extract --prune`.

**Upgrade note:** no existing index is invalidated by this release —
fingerprints stay byte-identical for corpora with no agent-authored
sidecar entries. An agent-authored transcript, once registered, is never
superseded by a later `pdf`-extra install (only `retrieval extract
--force` overwrites it, and only for PDFs). Adding the new agent-only
media suffixes to discovery does change the fingerprint for any corpus
that **contains** `.docx`/`.pptx`/`.xlsx`/image files under its indexed
root: such a corpus reindexes once, automatically, on the next
`index`/`query` after upgrading, picking up the newly discovered files as
`agent-only` stubs. A corpus with none of these file types is unaffected.
(Audio/video support landed in 0.11.0, below, via a third
agent-orchestrated media tier rather than the agent-only tier this release
introduced — an agent can read a docx or image natively, but not an mp4's
bytes, so it needed a different mechanism.)

## 0.8.0 — 2026-08-03

- **\[engine\] PDFs in the project tree are now indexed automatically**, via
  Markdown sidecar transcripts — no flag needed. New `retrieval.extractors`
  module (stdlib-only at import scope) writes each PDF's extracted text to
  `<root>/.agentic-retrieval/extracted/<rel-path>.md`, keyed by a
  content-hash + extractor-version manifest so re-extraction is a no-op on
  unchanged files across `index --auto`'s multiple loader passes. With the
  `pdf` extra installed (`pypdf`, included in the `all` extra), a PDF yields
  a real page-by-page transcript (dehyphenated, reflowed, running
  header/footer stripped); without it, a searchable placeholder stub is
  written instead, with a one-time `warning: pypdf is not installed` line on
  stderr per process — `index`/`query` never hard-fail on a missing `pypdf`
  backend. `.pdf` joined `project_loader.DEFAULT_EXTENSIONS`, with a
  two-stage discovery size cap (`extract_max_bytes`, 25MB, vs. the existing
  1MB plain-text cap) so real-world PDFs aren't silently excluded.
- **\[engine\]** New `--no-pdf` flag on `index` (opt-out of PDF
  auto-activation), persisted in the cache's `meta.loader_kw` and therefore
  sticky across later flag-less `index`/`query` calls.
- **\[engine\]** New `retrieval extract` subcommand pre-warms every PDF's
  sidecar transcript under a project root without touching any retriever
  index — `--force` re-extracts, `--prune` removes manifest entries/sidecar
  files for PDFs no longer present, `--json` emits a machine-readable
  summary. This is the *only* place PDF extraction hard-fails (a guidance
  `RuntimeError`) when the `pdf` extra isn't installed.
- **\[engine\] Fix:** the CLI's `loader_kw` (extensions/exclusions/size caps)
  is now actually threaded from `index`-time flags into fingerprinting and
  persisted cache metadata (`_loader_kw_from_args`, previously a stub
  returning `{}`) — a latent staleness bug where a loader-affecting flag
  wouldn't reliably trigger a rebuild.
- **\[engine\] Fix:** `load_chunk_documents`'s `Document.context` is now
  populated from each chunk's `heading` (previously left at its default),
  so the lexical retriever's citations/consolidated-hit `context` field
  carries a real breadcrumb (e.g. a PDF sidecar's `## Page N` heading)
  instead of being blank; `ContextualLexicalRetriever.index` no longer drops
  that `context` when rebuilding its enriched Documents.
- **\[engine\]** `index --retriever lexical+ctx` (one LLM call per chunk)
  now warns above 500 chunks and refuses outright above 2000 chunks unless
  `--allow-large-context` is passed, guarding against an accidentally
  expensive/slow index run on a large corpus.
- **Upgrade note:** a project containing PDFs will rebuild its index once on
  the first `index`/`query` call after upgrading (new discovery eligibility
  for `.pdf`, changed fingerprint); PDF-free projects are unaffected.

## 0.7.1 — 2026-07-27

- **\[docs\] Method explainers for every strategy.**
  `docs/concepts/retrieval-strategies.md` now explains each method, not
  just pi-serini: new "The lexical method" (TF-IDF + BM25 + RRF, and the
  contextual-enrichment link to Anthropic's Contextual Retrieval), "The
  turbovec method" (dense ANN over TurboQuant-quantized embeddings), "The
  hybrid method" (with the RRF citation — Cormack et al., SIGIR 2009), and
  "The tree-sitter method" (with the cAST citation — Zhang et al.,
  arXiv:2506.15655) sections; the hybrid-fusion how-to links to the new
  hybrid section.

## 0.7.0 — 2026-07-27

- **\[engine\]** `vendor/` added to `DEFAULT_EXCLUDE_DIRS` in
  `project_loader`, so vendored third-party code is skipped during file
  discovery like `node_modules` and `venv` already are.
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
- **\[docs\] Docs-only disambiguation.** The README, docs home, per-retriever
  how-to, troubleshooting table, the `lucene-retrieval-usage` skill, and the
  `PiSeriniRetriever` docstring now state explicitly that `pi-serini` (the
  strategy/registry key from the Pi-Serini paper) and `pyserini` (the
  Castorini library and install extra) are distinct names, not a typo for
  each other. No identifier, CLI flag, registry key, cache filename, class
  name, or extra name changed.
- **\[docs\] Pi-Serini method explainer.** `docs/concepts/retrieval-strategies.md`
  gains a "The pi-serini method" section with the full paper citation
  (Hsu, Yang, Lin — *Rethinking Agentic Search with Pi-Serini: Is Lexical
  Retrieval Sufficient?*, arXiv:2605.10848): the retrieve-deeper thesis,
  BrowseComp-Plus results, and the `k1=25, b=1` tuning rationale.

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
