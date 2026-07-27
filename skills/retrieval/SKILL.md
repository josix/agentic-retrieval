---
name: retrieval
description: Compare six retrieval strategies — contextual retrieval (TF-IDF/BM25/RRF fusion), its LLM-enriched lexical+ctx variant, turbovec (dense ANN), pi-serini (Lucene BM25), hybrid (lexical + dense fused with RRF), and tree-sitter (AST-boundary chunking with enclosing scope context) — with a stdlib-only offline core and graceful degradation when optional backends are missing. Use when the user wants to compare retrieval strategies on their project, search project files, or set up the retrieval engine.
trigger: /retrieval
---

# /retrieval

Compare six retrieval strategies on the invoking project's own files
(docs/code under the project root):

1. **Contextual retrieval** (`lexical`) — TF-IDF + BM25 fused with
   reciprocal-rank fusion. Fully offline, zero required dependencies.
2. **Contextual retrieval + LLM enrichment** (`lexical+ctx`) — the same
   lexical retriever over text enriched by an LLM (or heuristic)
   contextualizer that situates each chunk or document before indexing.
   Opt-in (costs LLM tokens); needs the `remote` extra + `ANTHROPIC_API_KEY`
   for LLM enrichment.
3. **turbovec** — dense ANN retrieval over embeddings, quantized with
   TurboQuant. Needs `sentence-transformers` + `turbovec`, installed by setup.
4. **pi-serini** — Lucene BM25 via Pyserini, the reference lexical retriever
   from the Pi-Serini paper. Needs `pyserini` (installed by setup) + a Java 21
   JDK. (pi-serini names the strategy; pyserini names the library — both
   spellings are correct.)
5. **hybrid** — lexical + dense arms over the same corpus, fused with RRF at
   search time. Needs the same extras as turbovec.
6. **treesitter** — the same lexical BM25+TF-IDF+RRF ranking over AST-boundary
   chunks (cAST), carrying an enclosing function/class breadcrumb on each hit.
   Needs the `treesitter` extra (only for chunking; ranking itself is
   zero-dependency).

The default `lexical` retriever runs fully offline with zero required
dependencies. Every optional strategy degrades gracefully — a missing
backend is skipped with a note, never a hard failure.

## Usage

```
/retrieval setup
/retrieval index
/retrieval query "your query here"
```

`index` with no `--retriever` builds every strategy's cache in one pass
(lexical always; turbovec/pi-serini/hybrid/treesitter when their extras are present,
else skipped with a note — never a hard failure). `query` with no
`--retriever` (the default, alias `--retriever all`) **consolidates every
available strategy into a single deduplicated, ranked, explainable list** —
a ranked set of entry points a following conversation/agent can explore and
verify from (not a finished answer — see "Presenting your answer"). Pass
`--retriever <name>` to query one strategy instead — see "Agent routing"
below for when a single strategy is still the right call.

## What You Must Do When Invoked

Follow these steps in order. Do not skip steps.

### Step 1 — Ensure uv is available, then sync the environment

`uv` is the hard requirement for this path. Guard for it, then `uv sync`
the engine's environment with every optional extra — `uv sync` is
idempotent, so it is always safe to re-run:

```bash
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

Standalone (outside the plugin, running from the repo root):

```bash
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
uv sync --project engine --extra all
```

Core (no extras) must always succeed. If the full `--extra all` sync fails
it is almost always the `pyserini` extra (needs a Java 21 JDK on `PATH`; all
extras also need network access). Sync core only to confirm the base engine
works, then re-run the full sync once the prerequisite is in place:

```bash
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}"
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine"             # core only
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all # retry full install
```

Report which extras synced and which failed **verbatim** from `uv`'s own
output — do not paraphrase which groups succeeded.

### Step 2 — Search the project's own files (zero deps, always works)

Capture the project root **before** invoking `uv run` — the skill's working
directory does not change, but `--root`/`RETRIEVAL_ROOT` must point at the
invoking project, not the plugin's `engine/`. Use the `retrieval` console
script (installed by `setup`'s `uv sync`) to build a persisted, on-disk index
once with `index`, then search it (as many times as you like) with `query`:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all \
  retrieval index --root "$PROJECT_ROOT" --auto

UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all \
  retrieval query "your query here" --root "$PROJECT_ROOT" --auto
```

### Hyperparameters are YOUR decision (made inline, at index/query time)

You — the invoking agent — decide the hyperparameter values and pass them
directly on the `index` and `query` commands. There is no separate tuning
step to run first. Precedence is **explicit flag > `--auto`-derived >
static default**, applied per flag, so you can take the auto-derived
baseline and override only the knobs you have a reason to set.

At **index time**, always pass `--auto` (as in the snippet above) so the
engine derives a corpus-calibrated baseline (tokenizer mode, chunk sizes,
BM25 `k1`/`b`, turbovec embedding model + `bit_width`, Lucene `k1`/`b`)
from measured corpus
signals instead of static defaults. Then layer explicit overrides for
anything you already know about the corpus or the coming queries:

| You know / observe | Pass on `index` |
| --- | --- |
| Corpus is mostly code, or mostly prose (overriding the auto guess) | `--tokenizer code` / `--tokenizer plain` |
| Queries will be long/verbose natural language | `--bm25-k1 1.8` (raise term saturation) |
| Chunk lengths are uniform / wildly varied | `--bm25-b 0.3` / `--bm25-b 1.0` |
| Functions or sections are being split mid-definition | `--code-chars` / `--target-chars` / `--ast-max-chars` (larger) |
| Recall matters more than memory (small corpus) | `--bit-width 4` |
| A specific embedding model fits the domain | `--embed-model <name>` |
| Long documents indexed as few big chunks | `--lucene-k1 25 --lucene-b 1` |

At **query time**, pass `--auto` to resolve `top_k` from the index's
persisted corpus stats, or an explicit `--top-k N` when you know how many
candidates you want (explicit wins; static default is 5). In consolidated
mode you also own the fusion: `--weights "name:w,..."` to bias retrievers
you trust for this question's shape (see the routing table), and `--pool N`
to deepen each retriever's candidate pool for recall-critical questions.

Your decisions persist: `index` records the resolved values in each
strategy's cache meta (`hyperparams` + `corpus_stats`, visible via
`retrieval stats`), later rebuilds — including query-triggered
auto-reindexes — reuse them rather than reverting to defaults, and passing
*different* values marks the cache stale and rebuilds with the new ones.
So decide once at index time, re-decide only when the corpus or your
retrieval quality observations change. To validate a decision
data-driven, `retrieval eval` (see `docs/how-to/evaluate-retrievers.md`)
compares configurations against labeled queries.

`index` with no `--retriever` (the default, equivalent to `--retriever all`)
builds every default strategy — `lexical` (TF-IDF + BM25 fused with RRF, the
zero-dependency baseline), `turbovec` (dense ANN; needs the turbovec + local
extras), `pi-serini` (Lucene BM25; needs the pyserini extra + Java 21),
`hybrid` (lexical + dense fused with RRF; needs the turbovec extras), and
`treesitter` (AST-boundary chunks with enclosing-scope context; needs the
treesitter extra) — each persisted as its own JSON cache slot under
`<project-root>/.agentic-retrieval` by default (override the cache
location with `RETRIEVAL_INDEX_DIR`), so they
coexist and never invalidate each other. Indexing is **chunk**-granularity
(`load_chunk_documents`), so each cache holds one entry per chunk span, not
per file. One line per strategy is printed
(`lexical: indexed N chunks`, etc.); any strategy whose extras are missing
prints `<name>: skipped (<reason>)` and the run still exits 0, since the
always-available `lexical` strategy succeeding is what matters for the
default run. Pass `--retriever lexical|turbovec|pi-serini|hybrid|treesitter` to build
just one strategy instead — in that single-strategy form a missing extra
**hard-fails** with a guidance `RuntimeError` (exit 1) naming the install
command, instead of being skipped; fall back to `--retriever lexical` in
that case. `query` loads the cache for whichever `--retriever` the agent
picked and searches it, printing one `source_path:start_line-end_line` span
per line (best match first, no scores) — pass `--json` for
`{"query": ..., "results": [{"docid", "path", "start_line", "end_line",
"rank"}, ...]}` instead. Turn a hit into exact file content with
`Read(path, offset=start_line, limit=end_line-start_line+1)` — the span
locates the lines for you, so open them directly instead of grepping to
find them, then read outward from there (see Step 3). If that strategy's cache is
missing, or the project's files have changed since it was built (detected by
a content fingerprint, no manual invalidation needed), `query`
auto-reindexes just that strategy before searching (also hard-failing on a
missing extra, exit 1 — `query` never silently falls back); pass
`--stale-ok` to search the stale cache anyway instead.
`retrieval stats --root "$PROJECT_ROOT"` reports every built cache slot's
location, chunk count, file count, creation time, and staleness without
searching.

### `query`'s default is consolidated (all strategies, one ranked list)

`query` with **no `--retriever` flag** (equivalent to `--retriever all`)
loads every available strategy's cache, searches each, and merges/fuses them
into one deduplicated, ranked, explainable list via
`retrieval.consolidation.consolidate` — same-file overlapping/adjacent spans
across retrievers (e.g. a `lexical` line-chunk and a `treesitter` AST-chunk
over the same function) merge into a single candidate. This is almost always
what you want by default: it removes the need to pick one `--retriever` up
front and surfaces cross-retriever agreement as a relevance signal. Text mode
prints one line per result — `path:start-end  [score=... agree=n/m
conf=high|medium|low  via a,b,c]  context` (first token stays `path:start-end`
so the `Read` affordance survives); `--json` emits `{"query", "mode":
"consolidated", "retrievers": [...], "skipped": [{"name", "reason"}, ...],
"results": [{"docid", "path", "start_line", "end_line", "rank", "context",
"score", "provenance", "agreement", "confidence", "contributors"}, ...]}`.
Pass `--output PATH` to also persist that same JSON envelope to disk (a
persistable ranking a following conversation/agent verifies and explores
from — see "Presenting your answer"), and
`--weights "name:w,..."` to bias specific retrievers in the fusion. A missing
backend is skipped (noted on stderr in text mode, in `"skipped"` in JSON),
never a hard failure, as long as the always-available `lexical` strategy
consolidates successfully (exit 0); exit 1 only if even `lexical` is
unusable. Full detail: `docs/how-to/consolidated-query.md`.

### Agent routing — when to query a single retriever instead

Pass `--retriever <name>` to skip consolidation and query exactly one
strategy — its output is byte-identical to the pre-consolidation CLI (plain
`path:start-end` lines, or the `{"query", "results": [...]}` JSON shape with
no `score`/`provenance` fields). Prefer this when you already know which
method fits the question's shape:

| Query characteristic | Query with |
| --- | --- |
| Exact keywords / identifiers / code tokens; zero-dep default | `--retriever lexical` |
| Paraphrase / synonyms / wording differs from documents | `--retriever turbovec` |
| Lucene-grade BM25 depth/scale needed | `--retriever pi-serini` |
| Uncertain — vocabulary mismatch vs genuine irrelevance | `--retriever hybrid` |
| Code/script; want AST-boundary spans + enclosing scope | `--retriever treesitter` |
| Chosen backend skipped/errored | fall back to `--retriever lexical` |

**Tradeoff — what single-retriever mode gives up.** Querying one strategy is
faster (one index load + one search, no fan-out) but forfeits the
cross-retriever *agreement* signal the consolidated default provides: with a
single arm there is no `agreement`/`confidence` and no `score`/`provenance`
in the output, so you lose the built-in corroboration that two independent
methods surfaced the same span. Prefer `--retriever <name>` when you already
know the query's shape or are latency-bound; keep the consolidated default
(`--retriever all`) when you want that agreement-as-relevance check.
Choosing the retriever is manual today — automatic query->retriever routing
is a future enhancement, and the eval harness (`retrieval eval`, see
`docs/how-to/evaluate-retrievers.md`) exists to make that decision
data-driven.

See `hybrid-retrieval-usage` for the full decision walkthrough.

For LLM or heuristic contextualization before indexing (closing
vocabulary-mismatch gaps), see `lexical-retrieval-usage` /
`hybrid-retrieval-usage`.

**Failure mode** — if `load_documents` finds no indexable files under the
project root, `index` persists an empty (zero-doc) cache rather than raising
or exiting, and `query` then returns no results. Check that
`--root`/`RETRIEVAL_ROOT`/`PROJECT_ROOT` points at the intended project
directory, and remember the default exclusions (`.git`, `.venv`,
`node_modules`, etc. — see
`retrieval/project_loader.py::DEFAULT_EXCLUDE_DIRS`) may have skipped
everything if it points at an already-excluded or empty directory.

### Step 3 — Explore from the hits, then answer

Retrieved spans are **seeds for exploration, not the final answer**. A hit
tells you *where* to start, not that you are done. Retrieval-assisted answers
must match or exceed what plain code-path tracing produces — the consolidated
list SEEDS the trace, it never caps it.

**Triage first (shallow vs deep).** For a bare locate-a-symbol ask ("where is
X defined", "which file holds Y"), read the top span, confirm it in the live
file, and answer — skip the phased loop. For any explanatory / how-does-X-work
/ "trace the flow" / comprehensiveness question, run the full loop below. When
in doubt, go deep.

Run these phases in order:

**Phase Q — Decompose.** Break the question into 3-6 sub-aspects, e.g. entry
point, data flow, core algorithm, edge cases / conflict handling, downstream
consumption. Write them down; they become the coverage checklist.

**Phase R — Retrieve per sub-question.** Run one `retrieval query` per
sub-aspect (iterate, don't cram one query). Persist each with
`--json --output <scratch>/aspect-N.json` so the seeds survive the trace.
A hit often reveals the codebase's real vocabulary — feed that back into a
fresh query for aspects the first wording missed.

**Phase T — Trace (MANDATORY, this is where detail comes from).** For every
seed span, do NOT stop at the snippet:

1. **Read the span in the live file** — widen the `Read` window past any
   mid-definition cut, and re-confirm it is current, not legacy/deprecated.
2. **Follow references outward** — chase callers, callees, imports, config
   keys, and cross-file mentions with Grep/Read. Walk the execution flow
   end-to-end: entry point → branches → core algorithm → terminal/consumption.
3. **Re-query with what you learned** — retrieve → read → refine → retrieve.

For a broad question (≥4 independent sub-aspects or a wide codebase), you MAY
fan out: spawn one `retrieval-tracer` subagent per sub-aspect via Task,
passing its `aspect-N.json` seeds. Tracers return DETAILED traces (file:line
+ key snippets), which you stitch together — they do not replace your own
synthesis. Default to sequential single-agent tracing when the question is
narrow.

**Phase C — Coverage gate (fill before you answer).** Do not write the answer
until every box is checked:

- [ ] Each sub-aspect from Phase Q has ≥1 answer with a verified `file:line`.
- [ ] The full execution path is traced entry → exit (not just the top hit).
- [ ] Current-vs-legacy is disambiguated for every cited span.
- [ ] Every cited detail is verified against the live file, not the cache.
- [ ] Cross-file references (callers/callees/config) followed ≥1 hop.

If any box is unchecked → loop back to Phase R (new vocabulary) or Phase T
(trace further). There is no "stop after the first read" signal here.

**Phase S — Synthesize.** See "Presenting your answer" below.

### Presenting your answer (retrieval enriches, it does not replace tracing)

The consolidated list is scaffolding for *your* exploration — entry points
and ranking hints — not the answer you hand the user. When you write the
final answer:

- **Answer in traced prose with verified `file:line` citations.** Trace the
  actual code path from the retrieved entry points; cite lines you have
  opened and read. Do NOT open your answer with (or paste) the raw
  `path:start-end  [score=… agree=… conf=… via …]` table — that ranking
  metadata is an internal relevance hint, not user-facing content.
- **Rank and confidence are not currency or authority.** `score`, `agree`,
  and `conf` measure how strongly retrievers matched your *query text* and
  agreed with each other — NOT that a span is current, canonical, or
  non-deprecated. A `conf=high` hit can point straight at legacy or
  deprecated code that two retrievers happened to both surface.
- **Verify each span against the live file before relying on it.** The cache
  is a snapshot; re-read the span in the current file and confirm it is the
  present implementation. Check for deprecation markers, newer sibling
  definitions, and whether the symbol is still referenced before quoting it.
- **Prefer what direct tracing finds over what retrieval ranked.** If reading
  the code path reveals an implementation detail retrieval missed, that
  detail wins. Retrieval points; the code decides.

#### When results look noisy or low-relevance

Distinct from a backend being skipped/errored (that is the routing-table
fallback and the Failure mode above), the retriever can run fine yet return
spans that don't answer the question. When top hits look off-topic:

- **Re-query with different vocabulary** — rephrase using terms you saw in the
  code/docs so far, or synonyms of your original query.
- **Switch retriever** — if `lexical` returns keyword-matched but irrelevant
  spans (vocabulary mismatch), try `--retriever turbovec` (semantic) or
  `--retriever hybrid` (fuses both); if a paraphrase query drifts, try
  `--retriever lexical` with the exact identifier instead.
- **Raise `--top-k`** — a relevant span may sit just below the default cut;
  ask for more candidates and skim them (pi-serini's premise is that depth
  compensates for lexical gaps).
- **Skim, don't trust blindly** — read the top few spans before concluding; a
  low-ranked hit can still be the right one. Retrieval ranks candidates, it
  does not decide relevance for you.
- **Fall back to direct navigation** — if retrieval keeps missing, grep for a
  known token or browse the likely directory directly. Retrieval augments your
  existing tools; it never replaces them.

### Advanced: direct engine API

The CLI above wraps the same `load_chunk_documents` -> `index` ->
`search_detailed` flow this heredoc form runs in-memory, per invocation, with
no persisted cache — useful for one-off scripting or when you need the
`SearchHit` objects (with file:line spans) directly rather than the CLI's
plain-text/JSON output:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
RETRIEVAL_ROOT="$PROJECT_ROOT" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import LexicalRetriever

docs = load_chunk_documents(os.environ["RETRIEVAL_ROOT"])
r = LexicalRetriever()
r.index(docs)
for hit in r.search_detailed("your query here", top_k=5):
    print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}")
PY
```

Standalone (outside the plugin, running from the repo root):

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
RETRIEVAL_ROOT="$PROJECT_ROOT" uv run --project engine --extra all python - <<'PY'
import os

from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import LexicalRetriever

docs = load_chunk_documents(os.environ["RETRIEVAL_ROOT"])
r = LexicalRetriever()
r.index(docs)
for hit in r.search_detailed("your query here", top_k=5):
    print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}")
PY
```

## Engine API — retrievers

The six retrieval methods are implemented as classes in
`engine/retrieval/retrievers.py`, all built over the shared
`engine/retrieval/document.py::Document` record
(`docid: str`, `text: str`, `url: str = ""`, `source_path: str = ""`,
`start_line: Optional[int] = None`, `end_line: Optional[int] = None`,
`context: str = ""`).
`load_chunk_documents(root)` (`engine/retrieval/project_loader.py`) is the
production loader: it chunks every discovered file and returns one
chunk-granularity `Document` per span, `docid` formatted `"{path}:{start}-
{end}"`. Every retriever's `search_detailed(query, top_k) -> List[SearchHit]`
resolves each ranked result's span **from that Document metadata**, never by
parsing the docid string — a `SearchHit` carries `docid`, `source_path`,
`start_line`, `end_line`, `rank`, `context` (an optional enclosing
function/class breadcrumb, populated by the tree-sitter retriever), so a
coding agent can turn a hit straight into `Read(hit.source_path,
offset=hit.start_line, limit=hit.end_line - hit.start_line + 1)`.
`search(query, top_k) -> List[str]` remains as a thin
docid-only projection of `search_detailed` for backward compatibility. Select
a class either directly or via `retrieval.retrievers.REGISTRY` /
`build_retriever(name)`.

| Method | Class | REGISTRY key | Optional extra | Always available? |
| --- | --- | --- | --- | --- |
| Contextual lexical retrieval (TF-IDF + BM25 + RRF) | `LexicalRetriever` | `lexical` | none (core) | Yes |
| Contextual lexical + LLM-enriched text | `ContextualLexicalRetriever` | `lexical+ctx` | `[remote]` for enrichment | Yes for base; needs `ANTHROPIC_API_KEY` for enrichment |
| turbovec (dense ANN) | `TurbovecRetriever` | `turbovec` | `[turbovec,local]` | No — `RuntimeError` if uninstalled |
| pi-serini (Lucene BM25) | `PiSeriniRetriever` | `pi-serini` | `[pyserini]` | No — `RuntimeError` if uninstalled or Java missing |
| hybrid (lexical + dense RRF fusion) | `HybridRetriever` | `hybrid` | `[turbovec,local]` | No — `RuntimeError` if uninstalled |
| tree-sitter (AST-boundary chunks + scope context) | `TreeSitterRetriever` | `treesitter` | `[treesitter]` (chunking only) | Yes for ranking; chunking needs `[treesitter]` or falls back |

```python
from retrieval.retrievers import build_retriever
from retrieval.document import Document

docs = [Document("d1", "Routers forward packets between networks.")]
r = build_retriever("lexical")
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
```

## Graceful degradation

- The core `uv sync` (no extras) must succeed — everything downstream
  depends on it.
- If the full `--extra all` sync fails, sync core only to confirm the base
  engine works, then re-run the full sync once the prerequisite (usually a
  Java 21 JDK for `pyserini`, or network access) is in place — see Step 1's
  core-only fallback.
- `TurbovecRetriever.index()` and `PiSeriniRetriever.index()` raise a
  `RuntimeError` with install instructions when their optional dependency is
  missing — catch this and fall back to `LexicalRetriever` rather than
  aborting.
- `load_ast_chunk_documents()` (feeding `TreeSitterRetriever`) raises the
  same style of `RuntimeError` when the treesitter extra is missing;
  `TreeSitterRetriever.index()` itself never does (it needs no optional
  dependency) — catch the loader's error and fall back to
  `load_chunk_documents()` + `LexicalRetriever`.
- The `lexical` retriever always runs — it is the zero-dependency baseline
  every other strategy is compared against.
- `retrieval index` (default, all-strategy mode) degrades gracefully: a
  missing backend prints `<name>: skipped (<reason>)` and the run still
  exits 0, as long as `lexical` itself built successfully.
- `retrieval index --retriever <name>` (single-strategy) and
  `retrieval query --retriever <name>` do **not** degrade — a missing
  backend hard-fails with a guidance `RuntimeError` and exit 1, with no
  silent fallback. If you want graceful behavior at query time, catch the
  failure yourself and retry with `--retriever lexical`.

## Guardrails

- **Cost warnings**: LLM-enriched indexing (`lexical+ctx`, or
  `retrieval.llm_contextualizer`) calls a network LLM and costs tokens — warn
  the user and prefer explicit opt-in before running it without being asked.
- **Anthropic-native default**: when a strategy needs an LLM (contextualizer),
  default to Anthropic's API (`ANTHROPIC_API_KEY`) unless the user asks for a
  different provider.
- **No global installs, no bare `python`/`pip`**: every dependency lives in
  the `uv`-managed environment for `engine/` (at
  `$UV_PROJECT_ENVIRONMENT`, default `$HOME/.cache/agentic-retrieval/uv-venv`)
  — always invoke the engine via `uv run --project <path-to-engine>`, never a
  bare `python`/`python3`/`pip install` call, which would bypass that
  isolation and resolve to a different (likely dependency-less) interpreter.

## Related knowledge skills

For per-method detail (strengths/weaknesses, setup, concrete index/search
snippets, graceful degradation) and for combining methods, see:

- `lexical-retrieval-usage` — contextual lexical retrieval (TF-IDF + BM25 + RRF)
- `dense-retrieval-usage` — turbovec dense ANN retrieval
- `lucene-retrieval-usage` — pi-serini Lucene BM25 retrieval
- `code-retrieval-usage` — tree-sitter AST-boundary chunking for code corpora
- `hybrid-retrieval-usage` — fusing methods and choosing between them
