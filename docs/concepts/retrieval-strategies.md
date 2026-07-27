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

## The lexical method

`LexicalRetriever` builds two classical sparse indexes over the same
corpus — TF-IDF and BM25 — and fuses their per-query rankings with
reciprocal rank fusion (RRF; see [the hybrid method](#the-hybrid-method)
for the citation). Both algorithms match on shared **tokens**, not
meaning: a document ranks highly only if it contains words the query
contains. TF-IDF rewards rare, discriminative terms; BM25 adds
term-frequency saturation and document-length normalization; fusing the
two rankings hedges each scorer's individual biases at zero extra
dependency cost. The whole pipeline is pure stdlib, which is why it is
the always-available baseline every other strategy degrades to.

Its known failure mode is vocabulary mismatch: a query saying "carries
data" never matches a document saying "forwards packets". The
**lexical+ctx** variant closes that gap at *index* time instead of
changing the ranking function — each document is prefixed with a short
generated context (topics and key entities) before TF-IDF/BM25 are fit,
following Anthropic's [Contextual
Retrieval](https://www.anthropic.com/news/contextual-retrieval) method.
See [contextual retrieval](contextual-retrieval.md) for the
heuristic-vs-LLM contextualizer trade-off.

## The turbovec method

`TurbovecRetriever` matches on **meaning** instead of shared tokens: each
document is embedded with a `sentence-transformers` model (default
`all-MiniLM-L6-v2`, d=384) and ranked by inner-product similarity, so
paraphrases and synonyms — lexical retrieval's blind spot — still match.
The index is built with
[turbovec](https://github.com/RyanCodrai/turbovec)'s TurboQuant
quantizer, which is **data-oblivious**: a fixed random rotation plus
per-coordinate calibration derived from math rather than learned from the
corpus. That gives two practical properties — no training phase and no
rebuild as documents are added (unlike FAISS IVF/PQ), and embeddings
compressed to 2–4 bits/dimension (up to 16× smaller than float32) while
length-renormalized scoring keeps inner-product estimates unbiased.

The recall ceiling is set by the embedder, not the quantizer: the default
model is small, free, and offline-installable; swap in a larger embedder
for paper-grade dense recall at the cost of that provider's API. The
trade-off in the other direction: exact keyword and identifier matches
(error codes, proper nouns) can rank lower than BM25 would rank them.

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

## The hybrid method

`HybridRetriever` runs the lexical and turbovec arms over the same corpus
and fuses their rankings at search time with reciprocal rank fusion:

> Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher.
> *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
> Learning Methods.* SIGIR 2009.

RRF scores each candidate by `sum(1 / (k + rank))` across every ranking
it appears in — it needs only rank positions, never the incomparable raw
scores of different scorers, which is what makes fusing a BM25 ranking
with a cosine-similarity ranking sound. A document that both arms rank
moderately well beats one that a single arm ranks highly, hedging each
method's characteristic failure mode (lexical's vocabulary mismatch,
dense's weak exact-identifier precision). See [hybrid
fusion](../how-to/hybrid-fusion.md) for usage, the `k` and `weights`
knobs, and consolidating more than two rankings.

## The tree-sitter method

`TreeSitterRetriever` changes the *chunking*, not the ranking. Files are
parsed with tree-sitter and split at AST node boundaries following the
cAST method:

> Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei,
> and Tongshuang Wu. *cAST: Enhancing Code Retrieval-Augmented Generation
> with Structural Chunking via Abstract Syntax Tree.* arXiv:2506.15655,
> 2025. <https://arxiv.org/abs/2506.15655>

The chunker greedily merges consecutive sibling nodes while their
combined non-whitespace character count fits a budget, recurses into
nodes too large to fit alone, and hard-splits only leaf nodes that still
don't fit — so a hit's span is a complete, syntactically coherent unit
(a function, a class, a block) instead of an arbitrary character window.
Every chunk also carries a dotted breadcrumb of its enclosing scopes
(e.g. `Bar.baz` for a method inside a class), which is prefixed into the
ranked text so a query can match on scope names alone. Ranking is then
the same TF-IDF + BM25 + RRF as the lexical method — tree-sitter is only
needed at chunking time, and it inherits lexical's vocabulary-mismatch
limitation.

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
