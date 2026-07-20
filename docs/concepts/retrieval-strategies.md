# Retrieval strategies compared

Five retrieval strategies, all run over the invoking project's own
docs/code: **lexical** retrieval (TF-IDF + BM25 fused with reciprocal-rank
fusion) is the zero-dependency baseline that always works; **lexical+ctx**
layers LLM (or heuristic) document enrichment on top of it; **turbovec**
(dense ANN over quantized embeddings) matches on meaning instead of shared
words, at the cost of needing an embedding model; **pi-serini** (Lucene BM25
via Pyserini) gives you a mature, tunable inverted index instead of this
repo's from-scratch TF-IDF/BM25, at the cost of a Java 21 JVM; **hybrid**
(`HybridRetriever`) runs lexical and turbovec arms over the same corpus and
fuses their rankings with RRF at search time, inheriting turbovec's extras
requirement.

## Selection table

| Situation | Use |
|---|---|
| Zero deps, or query/document vocabulary is well aligned | `lexical` |
| Query wording likely differs from document wording (paraphrase, synonyms) | `turbovec` |
| Need Lucene-grade BM25 depth/scale, or the pi-serini paper's tuning | `pi-serini` |
| Unsure which failure mode dominates, and two backends are installed | Fuse with RRF (see [hybrid fusion](../how-to/hybrid-fusion.md)) |

Repeated with the fusion and contextualization rows:

| Situation | Prefer |
|---|---|
| Zero deps, or vocabulary well aligned | `lexical` |
| Query wording likely differs from document wording | `turbovec` |
| Need Lucene-grade BM25 depth/scale, or pi-serini's `k1=25, b=1` | `pi-serini` |
| Uncertain which failure mode dominates, both backends installed | Fuse lexical + dense (or lexical + Lucene) with RRF |
| Sparse retrieval keeps missing a document for lack of shared vocabulary, but standing up dense/Lucene isn't worth it | Contextualize at index time instead |
| Any optional backend raises `RuntimeError` | Fall back to plain `lexical` |

## Graceful degradation rationale

The `lexical` strategy is the zero-dependency baseline: it always runs when
searching the invoking project's own files. `turbovec` and `pi-serini` are
opt-in comparison retrievers — if their extra isn't installed, calling
`.index()` raises a `RuntimeError` with install instructions rather than
crashing silently.

This is a deliberate design choice: every optional backend's `index()`
checks for its dependency lazily and raises a guidance `RuntimeError` with
the exact `uv pip install` command needed, instead of letting a raw
`ImportError` propagate. Callers are expected to catch that `RuntimeError`
and fall back to `LexicalRetriever` — which has zero optional dependencies
and therefore never raises — rather than aborting a comparison run
entirely. The same pattern applies to LLM contextualization (see
[contextual retrieval](contextual-retrieval.md)): a missing `anthropic`
package or unset `ANTHROPIC_API_KEY` raises a guidance error rather than a
silent no-op or a crash.

## Next steps

- [Use each retriever](../how-to/use-each-retriever.md)
- [Hybrid fusion](../how-to/hybrid-fusion.md)
- [Contextual retrieval](contextual-retrieval.md)
