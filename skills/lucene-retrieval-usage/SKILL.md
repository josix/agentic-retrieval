---
name: lucene-retrieval-usage
description: This skill should be used when indexing or searching a corpus with the pi-serini Lucene BM25 retriever, tuning BM25 k1/b parameters for long documents, handling missing Java 21 / pyserini dependencies, or when corpus scale demands a production-grade Lucene index.
---

# Lucene Retrieval Usage (pi-serini)

Lucene BM25 retrieval via `PiSeriniRetriever` (`retrieval/retrievers.py`),
backed by Pyserini/Anserini — the reference lexical retriever from
[pi-serini](https://github.com/justram/pi-serini) ("Rethinking Agentic Search
with Pi-Serini: Is Lexical Retrieval Sufficient?").

## What it is

`PiSeriniRetriever` builds an in-memory Lucene inverted index
(`pyserini.index.lucene.LuceneIndexer`) over the corpus and queries it with
`LuceneSearcher.set_bm25(k1, b)`. Pi-serini's thesis: a well-configured
lexical (BM25) retriever, given **sufficient retrieval depth**, can be
competitive with dense retrieval for agentic deep research — instead of
closing the vocabulary gap with embeddings, retrieve deeper and let a capable
LLM agent compensate by reading more candidates.

Pi-serini's BrowseComp-Plus-tuned defaults push BM25 far from its usual
settings: `k1=25` (vs. Lucene's usual ~0.9) nearly disables term-frequency
saturation, letting repeated query terms keep accumulating score, and `b=1`
applies full document-length normalization — tuned for long (~5,000-word)
documents. `PiSeriniRetriever.__init__(k1=0.9, b=0.4)` defaults to Pyserini's
general-purpose values; pass `k1=25, b=1` to reproduce the paper's tuning on
long-document corpora.

## When to use it (vs lexical / turbovec)

**Strengths:**
- Lucene-grade BM25 at scale — a mature, battle-tested inverted-index engine
  (this plugin's `LexicalRetriever` is a from-scratch TF-IDF/BM25
  implementation suited to small/medium corpora, not Lucene's throughput).
- No embedding model, no GPU, no vector index — "just a Lucene inverted
  index," per pi-serini's framing.
- Tunable BM25 parameters (`k1`, `b`) let retrieval depth compensate for
  vocabulary gaps instead of switching to dense retrieval.

**Weaknesses:**
- Needs a JVM — Pyserini wraps Anserini, which requires a **Java 21 JDK on
  `PATH`**. This is the heaviest external dependency of the three methods
  here (no JVM needed for lexical or turbovec).
- Still a lexical/token matcher at its core — same vocabulary-mismatch
  weakness as plain `LexicalRetriever`; pi-serini's answer is "retrieve
  deeper," not "match on meaning."
- In-memory index construction (`LuceneIndexer` over a temp dir) has more
  setup/teardown overhead than the pure-Python indexes.

**Use pi-serini when:** you need Lucene-grade BM25 (larger corpora, tuned
`k1`/`b`, reproducing the pi-serini paper's setup) and a Java 21 JDK is
available. Compare against `lexical-retrieval-usage` when you want a
zero-dependency baseline with no JVM requirement, and `dense-retrieval-usage`
when queries and documents are likely to use different wording entirely. See
`hybrid-retrieval-usage` for combining Lucene BM25 with dense rankings.

## Setup

Dependencies (`pyserini`) come from the `pyserini` extra, installed by the
plugin's one-shot sync:

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

**Requires a Java 21 JDK on `PATH`** independently of the Python install. If
Java 21 isn't available, the `pyserini` import itself will fail at index
time even if the pip package installed; check `java -version` first if the
sync "succeeds" but indexing still raises.

## How to index and search

Index the invoking project's own files — capture the project root **before**
invoking `uv run`, then feed it to the Python snippet via an env var:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import load_chunk_documents
from retrieval.retrievers import PiSeriniRetriever

docs = load_chunk_documents(os.environ["PROJECT_ROOT"])

r = PiSeriniRetriever()  # default k1=0.9, b=0.4; pass k1=25, b=1 for BCP tuning
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
The Lucene doc `id` is the chunk docid; `search_detailed` resolves each
hit's span from Document metadata built at index time, never by parsing the
Lucene hit's id string, so a hit turns straight into
`Read(hit.source_path, offset=hit.start_line, limit=hit.end_line -
hit.start_line + 1)`. Treat that span as a seed to read and explore from, not
the final answer — follow the references it surfaces outward and re-query
with the vocabulary a hit reveals; if the top spans look noisy, re-query,
switch retriever, or raise `--top-k` (see the `retrieval` skill's Step 3).

Or via the registry: `from retrieval.retrievers import build_retriever;
build_retriever("pi-serini")`.

## Graceful degradation

`PiSeriniRetriever.index()` first checks for `pyserini.index.lucene` and
`pyserini.search.lucene`; if either import fails (missing package, or
Pyserini installed but no compatible JVM found), it raises:

```
RuntimeError: pi-serini retriever needs the 'pyserini' extra and Java 21:
  uv pip install -e '.[pyserini]'   (and install a JDK 21)
```

Catch this `RuntimeError` and fall back to `LexicalRetriever` (always
available). Fix: re-run `UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine"
--extra all` (or install a Java 21 JDK if the sync succeeded but no JVM was
found). Do not call `.search()` before `.index()` succeeds; it raises
`RuntimeError("call index() before search()")`.

## Cross-links

- `lexical-retrieval-usage` — contextual lexical retrieval (zero-dep baseline
  and fallback)
- `dense-retrieval-usage` — turbovec dense ANN retrieval
- `hybrid-retrieval-usage` — fusing Lucene BM25 with dense rankings, and the
  method-selection decision table
