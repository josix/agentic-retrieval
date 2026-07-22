---
name: lexical-retrieval-usage
description: This skill should be used when indexing or searching a corpus with the contextual lexical retriever (TF-IDF + BM25 fused with reciprocal-rank fusion), optionally enriching document text with LLM-generated context before indexing, or when a zero-dependency offline search over project files is needed.
---

# Lexical Retrieval Usage

Contextual lexical retrieval: TF-IDF + BM25 fused with reciprocal-rank fusion
(RRF), with an optional LLM-enrichment step at index time. Pure stdlib at its
core — the zero-dependency baseline every other retrieval method in this
plugin is compared against.

## What it is

`LexicalRetriever` (`retrieval/retrievers.py`) builds two classical sparse
indexes over the same corpus — a `TfidfIndex` and a `BM25Index` — and fuses
their per-query rankings with RRF (`retrieval/fusion.py`). Both algorithms
match on shared **tokens**, not meaning: a document ranks highly only if it
contains words the query contains (or word forms close enough for the
tokenizer to treat as the same term).

`ContextualLexicalRetriever` (same module) is `LexicalRetriever` plus one
extra step at index time: each document's text is prefixed with a short
LLM-generated context (topics + key entities) before TF-IDF/BM25 are fit.
This closes the vocabulary-mismatch gap *before* ranking runs, rather than
changing the ranking function itself — see Anthropic's Contextual Retrieval.
A chunk-level equivalent that indexes the invoking project's own files
(`ContextualRetriever` in `retrieval/index.py`, driven by `load_chunks(root)`)
uses a cheap deterministic heuristic contextualizer
(`retrieval/contextualizer.py`) instead of an LLM call.

## When to use it (vs turbovec / pi-serini)

**Strengths:**
- Zero required dependencies — pure stdlib, always available, no GPU, no
  embedding model, no JVM.
- Deterministic and fast to build/query; ideal as a baseline or fallback.
- With LLM enrichment (`ContextualLexicalRetriever`), closes vocabulary gaps
  cheaply relative to standing up a dense index.

**Weaknesses:**
- Core `LexicalRetriever` cannot bridge a query/document vocabulary mismatch
  on its own (e.g. query says "carries data", document says "forwards
  packets" — no shared tokens, no match) unless enrichment is applied.
- No semantic similarity — synonyms and paraphrases are invisible unless the
  literal words appear somewhere in the indexed text.

**Use lexical retrieval when:** you need a zero-dependency, always-available
baseline; when documents and queries are likely to share vocabulary; or when
you want to close vocabulary gaps at index time (enrichment) instead of
switching ranking function (dense) or index backend (Lucene). Compare against
`dense-retrieval-usage` (turbovec) when queries and documents likely differ in
wording, and `lucene-retrieval-usage` (pi-serini) when you need Lucene-grade
BM25 depth at scale. See `hybrid-retrieval-usage` for combining lexical with
another method via fusion.

## Setup

`LexicalRetriever` itself needs zero extras, but the plugin installs every
strategy's dependencies in one pass so the shared `uv` environment stays
consistent across invocations:

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

`ContextualLexicalRetriever`'s enrichment step additionally needs
`ANTHROPIC_API_KEY` exported — the `remote` package it uses is already
included in `--extra all`.

## How to index and search

Index the invoking project's own files — capture the project root **before**
invoking `uv run`, then feed it to the Python snippet via an env var:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import LexicalRetriever

docs = load_chunk_documents(os.environ["PROJECT_ROOT"])

r = LexicalRetriever()
r.index(docs)
for hit in r.search_detailed("what carries data between networks", top_k=5):
    print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}")
PY
```

`load_chunk_documents` chunks every discovered file (`retrieval/chunker.py`)
and returns one chunk-granularity `Document` per span
(`docid = "{path}:{start}-{end}"`); it skips the same VCS/dependency/build
directories (`.git`, `.venv`, `node_modules`, etc.) and filename patterns
that look like secrets (`.env`, `*.pem`, `*credentials*`, etc.) as
`load_documents` — see `retrieval/project_loader.py`. Each `search_detailed`
result's `source_path`/`start_line`/`end_line` come straight from that span
metadata (never parsed back out of the docid string), so a coding agent can
turn a hit into `Read(hit.source_path, offset=hit.start_line,
limit=hit.end_line - hit.start_line + 1)`. Treat that span as a seed to read
and explore from, not the final answer — follow the references it surfaces
outward and re-query with the vocabulary a hit reveals; if the top spans
look noisy, re-query, switch retriever, or raise `--top-k` (see the
`retrieval` skill's Step 3).

For small ad-hoc corpora built in-memory (no project directory needed), the
engine API also accepts a plain list of `Document`s directly:

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
from retrieval.document import Document
from retrieval.retrievers import LexicalRetriever

docs = [
    Document("d1", "Routers forward packets between networks and carry data."),
    Document("d2", "Photosynthesis converts sunlight into chemical energy in plants."),
]

r = LexicalRetriever()
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
PY
```

With LLM enrichment (costs one API call per document — warn the user before
running; needs the `remote` group + `ANTHROPIC_API_KEY`):

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
from retrieval.document import Document
from retrieval.retrievers import ContextualLexicalRetriever

docs = [Document("d1", "Routers forward packets between networks and carry data.")]
r = ContextualLexicalRetriever()  # lazily builds LLMDocumentContextualizer
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
PY
```

Or via the registry: `from retrieval.retrievers import build_retriever;
build_retriever("lexical")` / `build_retriever("lexical+ctx")`.

## Customizing file discovery

`discover_files` / `load_documents` / `load_chunks`
(`retrieval/project_loader.py`) all accept the same keyword-only overrides —
`extensions`, `exclude_dirs`, `exclude_globs`, `include_basenames`,
`max_bytes` — extend the matching `DEFAULT_*` constant with `|` rather than
replacing it outright, so you keep the built-in secret/VCS/build exclusions.
Runnable examples for each override: `references/file-discovery.md`.

## How to index chunk-level (feeds `ContextualRetriever`)

`load_chunks` discovers files, chunks each one (`retrieval/chunker.py`), and
returns a flat `List[Chunk]` — the input `ContextualRetriever`
(`retrieval/index.py`) expects, as opposed to the whole-document `Document`
list `LexicalRetriever` / `TurbovecRetriever` / `PiSeriniRetriever` expect.
Full snippet, and the
`contextualizer=` injection point (see `hybrid-retrieval-usage` for LLM
wiring): `references/chunk-level-indexing.md`.

## Graceful degradation

`LexicalRetriever` has no optional dependency — it never fails to build.
`ContextualLexicalRetriever.index()` lazily imports
`retrieval.llm_contextualizer.LLMDocumentContextualizer`, which raises a
`RuntimeError` ("LLM contextualizer needs the 'anthropic' package +
ANTHROPIC_API_KEY") if the `anthropic` package or API key is missing. On that
`RuntimeError`, fall back to plain `LexicalRetriever` — it always works. Fix:
re-run `UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all` and
resolve any error it reports.

## Cross-links

- `dense-retrieval-usage` — turbovec dense ANN retrieval
- `lucene-retrieval-usage` — pi-serini Lucene BM25 retrieval
- `hybrid-retrieval-usage` — fusing lexical with dense/Lucene, and the
  method-selection decision table
