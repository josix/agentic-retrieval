# Hybrid fusion

`retrieval/fusion.py::reciprocal_rank_fusion(rankings, k=60, weights=None)`
fuses two or more ranked lists (lists of **integer indices**, best first) by
`sum(weight_i / (k + rank + 1))` across every list a candidate appears in.
Raise `k` to flatten the influence of rank position; lower it to weight top
ranks more heavily. `weights` is an optional, same-length list of per-ranking
multipliers — `None` (the default) weights every ranking `1.0`, which is
numerically identical to the unweighted fusion every existing caller already
relies on.

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJECT_ROOT="$PROJECT_ROOT" uv run --project engine --extra all python - <<'PY'
import os

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.project_loader import load_documents
from retrieval.retrievers import LexicalRetriever, TurbovecRetriever

docs = load_documents(os.environ["PROJECT_ROOT"])
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

!!! warning "Integer-index remap caveat"
    `reciprocal_rank_fusion` doesn't know about docids — every ranking must
    be remapped into the same integer index space first (as above) before
    fusing, then mapped back for the final output.

## Method-selection table

| Situation | Prefer |
|---|---|
| Zero deps, or vocabulary well aligned | `lexical` |
| Query wording likely differs from document wording | `turbovec` |
| Need Lucene-grade BM25 depth/scale, or pi-serini's `k1=25, b=1` | `pi-serini` |
| Uncertain which failure mode dominates, both backends installed | Fuse lexical + dense (or lexical + Lucene) with RRF |
| Sparse retrieval keeps missing a document for lack of shared vocabulary, but standing up dense/Lucene isn't worth it | Contextualize at index time instead (see [LLM contextualization](llm-contextualization.md)) |
| Any optional backend raises `RuntimeError` | Fall back to plain `lexical` |

Full detail: `skills/hybrid-retrieval-usage/SKILL.md`.

## Consolidating more than two rankings

`reciprocal_rank_fusion` above requires remapping every ranking's docids into
a shared integer space by hand, and treats two overlapping-but-not-identical
spans (e.g. a `lexical` line-chunk and a `treesitter` AST-chunk over the same
function) as unrelated candidates. `retrieval.consolidation.consolidate`
wraps the same weighted RRF with span-aware deduplication across any number
of retrievers, and returns a ranked list that already carries provenance,
agreement, and a confidence label — this is what `retrieval query` (the
default, or `--retriever all`) runs under the hood. See
[Consolidated query](consolidated-query.md) and the [consolidation
API](../reference/api/consolidation.md).

## Next steps

- [Consolidated query (the default)](consolidated-query.md)
- [Retrieval strategies compared](../concepts/retrieval-strategies.md)
- [Reference: fusion API](../reference/api/fusion.md)
- [Reference: consolidation API](../reference/api/consolidation.md)
