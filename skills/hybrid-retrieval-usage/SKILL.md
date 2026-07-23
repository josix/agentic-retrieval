---
name: hybrid-retrieval-usage
description: This skill should be used when combining lexical, dense, and Lucene retrieval rankings via reciprocal rank fusion, choosing which single retrieval method fits a task, deciding whether to add LLM/heuristic contextualization before indexing, or when a single retrieval method returns unsatisfying results and rankings should be combined.
---

# Hybrid Retrieval Usage

Combine two or more retrieval methods with reciprocal rank fusion (RRF), and
decide when a single method is enough versus when to fuse or contextualize.

## What it is

`retrieval/fusion.py::reciprocal_rank_fusion(rankings, k=60)` takes a list of
ranked lists (each an ordered list of document indices, best first) and
returns a single fused ranking, scored by `sum(1 / (k + rank + 1))` across all
lists a document appears in. `LexicalRetriever` already uses this internally
to fuse its own TF-IDF and BM25 rankings; the same function fuses rankings
**across** retrievers — e.g. `LexicalRetriever` + `TurbovecRetriever` — to get
the benefit of both token-matching and semantic-matching signals in one
ranked list.

The same lexical + dense fusion also ships prepackaged as a first-class
strategy: `retrieval.retrievers.HybridRetriever` (REGISTRY key `hybrid`,
built via `build_retriever("hybrid")`) indexes a `LexicalRetriever` and a
`TurbovecRetriever` over the same corpus and fuses their rankings with RRF at
search time — this is what `retrieval index`/`query --retriever hybrid`
use. It needs the same `turbovec` + `local` extras as `TurbovecRetriever`
(indexing raises their guidance `RuntimeError` when absent). Use the class
when you want the standard lexical+dense pairing; use the manual
`reciprocal_rank_fusion` recipe below when fusing a different pair (e.g.
lexical + Lucene) or more than two arms.

Contextualization is a complementary, index-time lever (not a ranking-time
one): `retrieval.contextualizer` (deterministic, offline, zero cost) and
`retrieval.llm_contextualizer` (LLM-based, costs tokens) both close
vocabulary gaps by enriching the *indexed text*, before any retriever — lexical,
dense, or hybrid — ever ranks it.

## Method-selection decision table

| Situation | Prefer |
| --- | --- |
| Zero dependencies required, or corpus/query vocabulary is well aligned | `lexical` (`LexicalRetriever`) — see `lexical-retrieval-usage` |
| Query wording likely differs from document wording (paraphrase, synonyms) | `turbovec` (`TurbovecRetriever`) — see `dense-retrieval-usage` |
| Need Lucene-grade BM25 depth/scale, or reproducing pi-serini's tuned `k1=25, b=1` | `pi-serini` (`PiSeriniRetriever`) — see `lucene-retrieval-usage` |
| Uncertain which failure mode dominates (vocabulary mismatch vs. genuine irrelevance), and both a lexical and a dense/Lucene backend are installed | **Hybrid**: fuse lexical + dense (or lexical + Lucene) rankings with RRF |
| Sparse retrieval keeps missing a document because it lacks the query's words, but standing up dense/Lucene isn't worth it | Add contextualization at index time (`retrieval.contextualizer` for free, `retrieval.llm_contextualizer` for a paid LLM call) instead of switching retrievers |
| Any optional backend (`turbovec`, `pi-serini`, or LLM contextualizer) raises `RuntimeError` | Fall back to plain `lexical` — it is the only method with no optional dependency |

`retrieval index` (default, no `--retriever`) builds all five cache slots in
one pass — lexical, turbovec, pi-serini, hybrid, and treesitter — so the
coding agent can query whichever method fits each question via `retrieval
query --retriever <name>` without a separate index run per method. If the
retriever a query picks turns out to be missing/skipped, fall back to
`--retriever lexical`. `retrieval query` **with no `--retriever` flag**
(default, alias `--retriever all`) skips this per-method choice entirely: it
consolidates every available strategy's ranking into one deduplicated,
explainable list in a single call — see "Consolidating more than two
rankings" below and `docs/how-to/consolidated-query.md`.

## Setup

Fusing methods needs every method's dependencies installed — one sync
covers all of them:

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

If the sync fails (e.g. `pyserini` needs a Java 21 JDK for lexical+pi-serini,
or network access for any extra), resolve the error `uv` reports and re-run
the same command.

## How to fuse two retrievers' rankings

Index the invoking project's own files — capture the project root **before**
invoking `uv run`, then feed it to the Python snippet via an env var:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import LexicalRetriever, TurbovecRetriever

docs = load_chunk_documents(os.environ["PROJECT_ROOT"])
query = "what carries data between networks"

lexical = LexicalRetriever()
lexical.index(docs)
lex_docids = lexical.search(query, top_k=10)

try:
    dense = TurbovecRetriever()
    dense.index(docs)
    dense_docids = dense.search(query, top_k=10)
except RuntimeError as exc:
    print(f"[skip] turbovec unavailable, falling back to lexical only: {exc}")
    dense_docids = []

# reciprocal_rank_fusion operates on integer indices, not docids directly —
# map each ranking's docids into a shared integer space, fuse, map back.
all_docids = sorted(set(lex_docids) | set(dense_docids))
to_idx = {docid: i for i, docid in enumerate(all_docids)}
from_idx = {i: docid for docid, i in to_idx.items()}

rankings = [[to_idx[d] for d in lex_docids]]
if dense_docids:
    rankings.append([to_idx[d] for d in dense_docids])

fused = reciprocal_rank_fusion(rankings)
print([from_idx[idx] for idx, _score in fused])
PY
```

`load_chunk_documents` (and `discover_files`/`load_documents`/`load_chunks`)
skip VCS/dependency/build directories and secret-looking filenames by
default, and accept `extensions`/`exclude_dirs`/`exclude_globs`/
`include_basenames`/`max_bytes` overrides — see
`lexical-retrieval-usage/references/file-discovery.md`. Each docid printed
above is a chunk span (`"{path}:{start}-{end}"`); prefer
`retriever.search_detailed(...)` over the manual fusion recipe above when you
just need `SearchHit`s (with `source_path`/`start_line`/`end_line` already
resolved) instead of raw docids — `HybridRetriever`/`build_retriever("hybrid")`
does exactly this fusion internally. Treat a resolved span as a seed to read
and explore from, not the final answer — follow the references it surfaces
outward and re-query with the vocabulary a hit reveals; if the top spans
look noisy, re-query, switch retriever, or raise `--top-k` (see the
`retrieval` skill's Step 3).

## Consolidating more than two rankings

The manual `reciprocal_rank_fusion` recipe above works for two arms and
requires you to remap docids into a shared integer space yourself. For three
or more arms — or whenever you want span-aware deduplication (a `lexical`
line-chunk and a `treesitter` AST-chunk over the same function should count
as *one* candidate, not two), plus provenance/agreement/confidence baked into
the result — use `retrieval.consolidation.consolidate` instead:

```python
from retrieval.consolidation import consolidate

per_retriever_hits = {
    "lexical": lexical.search_detailed(query, top_k=15),
    "turbovec": dense.search_detailed(query, top_k=15),
    "treesitter": treesitter.search_detailed(query, top_k=15),
}
ranked = consolidate(per_retriever_hits)  # List[ConsolidatedHit], best first
```

This is exactly what `retrieval query` (no `--retriever`, or `--retriever
all`) does under the hood — see `docs/how-to/consolidated-query.md` for the
CLI form (including `--weights` and `--output`) and
`docs/reference/api/consolidation.md` for the full API.

## Provider selection (experimental)

`retrieval/providers.py::get_contextualizer()` picks a contextualizer via the
`RAG_PROVIDER` env var, defaulting to the fully offline `"heuristic"`
provider — the other three choices (`"ollama"`, `"remote"`,
`"sentence_transformers"`) are stub wiring points that always raise
`RuntimeError`, not working integrations yet. Full detail and the
don't-rely-on-stubs caveat: `references/providers.md`.

## Contextualization boost (index-time, not ranking-time)

Before choosing hybrid fusion, consider whether contextualizing the indexed
text closes the gap more cheaply:

- `retrieval.contextualizer.make_context` / `contextualize` — deterministic,
  offline, zero cost. Prepends a structural breadcrumb (`[title > heading]` +
  heading tokens + previous chunk's first sentence) to each chunk. Always
  available, no setup beyond the base install.
- `retrieval.llm_contextualizer.LLMContextualizer` /
  `LLMDocumentContextualizer` — an LLM reads the whole document and writes a
  1-2 sentence context per chunk (or per document, via
  `ContextualLexicalRetriever` — see `lexical-retrieval-usage`). **Needs the
  `remote` group installed by setup + `ANTHROPIC_API_KEY`, and costs one LLM
  call per chunk/document — warn the user before running this, and prefer
  explicit opt-in.** The
  module-level `retrieval.llm_contextualizer.contextualize_llm(chunk,
  all_chunks_in_doc) -> str` is a one-liner convenience over a lazily-built
  default `LLMContextualizer` — same signature as
  `retrieval.contextualizer.contextualize`, so it drops straight into any
  `(chunk, doc_chunks) -> str` slot (e.g. `ContextualRetriever.build`'s
  `contextualizer=` below).

Contextualization and hybrid fusion are complementary, not exclusive: you can
contextualize the text fed into `LexicalRetriever` *and* fuse its ranking with
`TurbovecRetriever`'s.

### Wiring a custom contextualizer into `ContextualRetriever`

`ContextualRetriever.build(chunks, use_context=True, contextualizer=...)`
accepts an injected `(chunk, doc_chunks) -> str` callable — the heuristic by
default, or an LLM-based one (costs one call per chunk; needs the `remote`
group installed by setup + `ANTHROPIC_API_KEY` — warn the user and prefer
opt-in). Full snippets for both the LLM and heuristic wiring:
`references/custom-contextualizer.md`.

## Graceful degradation

Every optional backend used in a fusion (`TurbovecRetriever`,
`PiSeriniRetriever`, `LLMContextualizer`/`LLMDocumentContextualizer`) raises a
`RuntimeError` with install instructions when its dependency is missing —
catch it, log a `[skip] <name>: <reason>` note, and fuse over whatever
rankings succeeded (or fall back to plain `lexical` if only one ranking is
available). Never let a missing optional backend abort the whole comparison.
Fix: re-run `UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all`
(plus install a Java 21 JDK for pi-serini) and resolve any error it reports.

## Cross-links

- `lexical-retrieval-usage` — contextual lexical retrieval (the always-on
  baseline every hybrid fusion includes)
- `dense-retrieval-usage` — turbovec dense ANN retrieval
- `lucene-retrieval-usage` — pi-serini Lucene BM25 retrieval
- `code-retrieval-usage` — tree-sitter AST-boundary chunking for code corpora
- `docs/how-to/consolidated-query.md` — the default `retrieval query`
  consolidated-ranking mode (`retrieval.consolidation.consolidate`)
