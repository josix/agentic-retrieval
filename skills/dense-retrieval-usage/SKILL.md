---
name: dense-retrieval-usage
description: This skill should be used when indexing or searching a corpus with the turbovec dense ANN retriever, choosing quantized embedding-based retrieval over lexical matching, handling the RuntimeError raised when the turbovec + local extras are not installed, or when keyword search misses paraphrases and synonyms (vocabulary-mismatch problems).
---

# Dense Retrieval Usage (turbovec)

Dense approximate-nearest-neighbor (ANN) retrieval over quantized embeddings,
via `TurbovecRetriever` (`retrieval/retrievers.py`), backed by
[turbovec](https://github.com/RyanCodrai/turbovec)'s TurboQuant quantizer.

## What it is

`TurbovecRetriever` embeds each document with a `sentence-transformers` model
(default `all-MiniLM-L6-v2`, d=384), builds a `TurboQuantIndex` over the
resulting vectors, and ranks by inner-product similarity on the quantized
vectors. TurboQuant is a **data-oblivious** quantizer — a fixed random
rotation plus per-coordinate calibration, derived from math rather than
learned from the corpus, so there is **no training phase and no rebuild** as
the corpus grows (`index.add(vectors)` is enough). It compresses embeddings to
2-4 bits/dimension (up to 16x smaller than float32) while its length-
renormalized scoring keeps inner-product estimates unbiased at zero
search-time cost.

## When to use it (vs lexical / pi-serini)

**Strengths:**
- Matches on **meaning**, not shared tokens — bridges vocabulary mismatch
  between query and document wording (the failure mode plain lexical
  retrieval cannot fix without enrichment).
- Memory-cheap relative to uncompressed dense retrieval (turbovec's whole
  premise: float32 dense indexes are RAM-bound; TurboQuant is not).
- Online ingest — no retrain/rebuild step as documents are added, unlike
  FAISS IVF/PQ.

**Weaknesses:**
- Needs an embedding model and the `turbovec` + `sentence-transformers`
  packages — not zero-dependency.
- Recall ceiling is set by the embedder, not by TurboQuant itself: the
  default `all-MiniLM-L6-v2` (d=384) is a small, free, offline-installable
  model — swap in a larger embedder (e.g. an OpenAI d=1536/3072 model) for
  paper-grade dense recall, at the cost of needing that provider's API.
- No lexical precision boost — exact keyword/identifier matches (error
  codes, proper nouns) can rank lower than a BM25 index would rank them.

**Use dense retrieval when:** queries are likely to paraphrase or use
different words than the documents (synonyms, indirect phrasing), and you can
afford an embedding model plus the `turbovec` + `local` extras. Compare against
`lexical-retrieval-usage` when the corpus is small/vocabulary-aligned or you
need a zero-dependency baseline, and `lucene-retrieval-usage` when you want
Lucene-grade lexical depth instead of embeddings. See `hybrid-retrieval-usage`
to fuse dense with lexical rankings via RRF.

## Setup

Dependencies (`turbovec`, `sentence-transformers`) come from the `turbovec`
and `local` extras, both installed by the plugin's one-shot sync:

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

No global installs. If the sync fails (usually a network/pip-index issue),
resolve the error `uv` reports and re-run the same command.

## How to index and search

Index the invoking project's own files — capture the project root **before**
invoking `uv run`, then feed it to the Python snippet via an env var:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import TurbovecRetriever

docs = load_chunk_documents(os.environ["PROJECT_ROOT"])

r = TurbovecRetriever()  # default: model_name="sentence-transformers/all-MiniLM-L6-v2", bit_width=4
r.index(docs)
for hit in r.search_detailed("what carries data between networks", top_k=5):
    print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}")
PY
```

`load_chunk_documents` chunks every discovered file and returns one
chunk-granularity `Document` per span (`docid = "{path}:{start}-{end}"`),
skipping VCS/dependency/build directories and secret-looking filenames by
default — see `lexical-retrieval-usage` for the full exclusion list and a
minimal inline-`Document` example, and
`lexical-retrieval-usage/references/file-discovery.md` for overriding
`extensions`/`exclude_dirs`/`exclude_globs`/`include_basenames`/`max_bytes`.
Each `search_detailed` result's span comes from that Document metadata, so a
hit turns straight into `Read(hit.source_path, offset=hit.start_line,
limit=hit.end_line - hit.start_line + 1)`. Treat that span as a seed to read
and explore from, not the final answer — follow the references it surfaces
outward and re-query with the vocabulary a hit reveals; if the top spans
look noisy, re-query, switch retriever, or raise `--top-k` (see the
`retrieval` skill's Step 3).

`TurbovecRetriever.__init__(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", bit_width: int = 4)`
takes both as explicit constructor args — swap in a larger embedder (any
`sentence-transformers`-compatible model name) and/or raise `bit_width` for
higher-fidelity quantization at the cost of memory:

```python
r = TurbovecRetriever(model_name="sentence-transformers/all-mpnet-base-v2", bit_width=8)
```

Or via the registry: `from retrieval.retrievers import build_retriever;
build_retriever("turbovec")`.

## Graceful degradation

`TurbovecRetriever.index()` first checks for the `sentence_transformers` and
`turbovec` packages; if either is missing, it raises:

```
RuntimeError: turbovec retriever needs the 'turbovec' + 'local' extras:
  uv pip install -e '.[turbovec,local]'
```

Catch this `RuntimeError` and fall back to `LexicalRetriever` (always
available) — never let missing `turbovec` + `local` extras abort a comparison run. Fix:
re-run `UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all`
and resolve any error it reports. Do not call `.search()` before `.index()`
succeeds; it raises `RuntimeError("call index() before search()")`.

## Cross-links

- `lexical-retrieval-usage` — contextual lexical retrieval (zero-dep baseline
  and fallback)
- `lucene-retrieval-usage` — pi-serini Lucene BM25 retrieval
- `code-retrieval-usage` — tree-sitter AST-boundary chunking for code corpora
- `hybrid-retrieval-usage` — fusing dense with lexical, and the
  method-selection decision table
