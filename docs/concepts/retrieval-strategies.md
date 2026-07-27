# Retrieval strategies compared

Six retrieval strategies, all run over the invoking project's own
docs/code: **lexical** retrieval (TF-IDF + BM25 fused with reciprocal-rank
fusion) is the zero-dependency baseline that always works; **lexical+ctx**
layers LLM (or heuristic) document enrichment on top of it; **turbovec**
(dense ANN over quantized embeddings) matches on meaning instead of shared
words, at the cost of needing an embedding model; **pi-serini** (Lucene BM25
via Pyserini) gives you a mature, tunable inverted index instead of this
repo's from-scratch TF-IDF/BM25, at the cost of a Java 21 JVM; **hybrid**
(`HybridRetriever`) runs lexical and turbovec arms over the same corpus and
fuses their rankings with RRF at search time, inheriting turbovec's extras
requirement; **tree-sitter** (`TreeSitterRetriever`) runs the same lexical
ranking over AST-boundary ("cAST") code chunks, carrying an enclosing
function/class breadcrumb on every hit — the chunking step needs the
`treesitter` extra, but the ranker itself has zero optional dependencies.

## Selection table

| Situation | Use |
|---|---|
| Zero deps, or query/document vocabulary is well aligned | `lexical` |
| Query wording likely differs from document wording (paraphrase, synonyms) | `turbovec` |
| Need Lucene-grade BM25 depth/scale, or the pi-serini paper's tuning | `pi-serini` |
| Unsure which failure mode dominates, and two backends are installed | Fuse with RRF (see [hybrid fusion](../how-to/hybrid-fusion.md)) |
| Code/script corpus; want AST-boundary spans + enclosing scope context | `treesitter` |

Repeated with the fusion and contextualization rows:

| Situation | Prefer |
|---|---|
| Zero deps, or vocabulary well aligned | `lexical` |
| Query wording likely differs from document wording | `turbovec` |
| Need Lucene-grade BM25 depth/scale, or pi-serini's `k1=25, b=1` | `pi-serini` |
| Uncertain which failure mode dominates, both backends installed | Fuse lexical + dense (or lexical + Lucene) with RRF |
| Sparse retrieval keeps missing a document for lack of shared vocabulary, but standing up dense/Lucene isn't worth it | Contextualize at index time instead |
| Any optional backend raises `RuntimeError` | Fall back to plain `lexical` |

## The pi-serini method

The `pi-serini` strategy reproduces the reference lexical retriever from
the Pi-Serini paper:

> Tz-Huan Hsu, Jheng-Hong Yang, and Jimmy Lin. *Rethinking Agentic Search
> with Pi-Serini: Is Lexical Retrieval Sufficient?* arXiv:2605.10848, 2026.
> <https://arxiv.org/abs/2605.10848> — code:
> <https://github.com/justram/pi-serini>

The paper asks whether dense (embedding-based) retrieval is still needed
for agentic deep research now that LLM agents reason and use tools well.
Its answer: a well-configured lexical (BM25) retriever with **sufficient
retrieval depth** is enough. Dense retrieval exists mostly to close the
vocabulary gap between query and document wording; Pi-Serini's reframing
is that the retriever no longer has to solve that gap — retrieve deeper
and let a capable LLM agent compensate by reading more candidates. The
system itself is a search agent with three tools (BM25 document retrieval
over a Lucene inverted index via the Pyserini library, web browsing, and
document reading) — no embedding model, no GPU, no vector index.

On the BrowseComp-Plus benchmark, Pi-Serini paired with a frontier LLM
reached 83.1% answer accuracy and 94.7% surfaced-evidence recall,
surpassing released search agents built on dense retrievers. Ablations
show neither ingredient is optional: BM25 parameter tuning alone added
18.0% accuracy and 11.1% evidence recall, and deeper retrieval added
25.3% evidence recall.

The tuning is far from BM25's textbook settings: `k1=25` (vs. Lucene's
usual ~0.9) nearly disables term-frequency saturation so repeated query
terms keep accumulating score, and `b=1` applies full document-length
normalization — both tuned for the benchmark's long (~5,000-word)
documents. This repo's `PiSeriniRetriever` defaults to Pyserini's
general-purpose `k1=0.9, b=0.4`; pass `k1=25, b=1` to reproduce the
paper's setup. The trade-off is unchanged at its core: BM25 remains a
token matcher, so the method's answer to vocabulary mismatch is "retrieve
deeper," not "match on meaning" — see the selection tables above for when
`turbovec` or hybrid fusion fits better.

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
