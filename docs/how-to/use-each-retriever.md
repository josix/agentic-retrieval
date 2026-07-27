# Use each retriever

All six retriever classes share the same tiny contract: `index(documents)`
then `search(query, top_k) -> List[str]` (ranked docids). `Document` is
`(docid: str, text: str, url: str = "")`. This page walks through the three
base backends — `lexical`, `turbovec`, and `pi-serini`; for
`ContextualLexicalRetriever` see
[LLM contextualization](llm-contextualization.md), for `HybridRetriever` see
[hybrid fusion](hybrid-fusion.md), and for `TreeSitterRetriever` see
`skills/code-retrieval-usage/SKILL.md`.

## Lexical — `LexicalRetriever` (always works)

TF-IDF + BM25 fused with RRF, document-level. Zero dependencies — use it as
the default and as the fallback whenever another backend is unavailable.

Setup: nothing beyond the core `uv sync`.

```bash
uv run --project engine --extra all python - <<'PY'
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

Verified output: `['d1', 'd2']`.

`LexicalRetriever` has no optional dependency, so it never raises — it's the
retriever every graceful-degradation fallback lands on. Details, the
project-loader-based snippet, and the LLM-enrichment variant
(`ContextualLexicalRetriever`) live in
`skills/lexical-retrieval-usage/SKILL.md`.

## turbovec — `TurbovecRetriever` (extras: `turbovec` + `local`)

Embeds documents with `sentence-transformers` and ranks by inner-product
similarity over TurboQuant-quantized vectors. Use it when queries and
documents are likely to use different words for the same idea.

Setup: synced by the full `uv sync --project engine --extra all`. If it was
skipped, re-run it.

```bash
uv run --project engine --extra all python - <<'PY'
from retrieval.document import Document
from retrieval.retrievers import TurbovecRetriever

docs = [Document("d1", "Routers forward packets between networks and carry data.")]
r = TurbovecRetriever()  # default: model_name="sentence-transformers/all-MiniLM-L6-v2", bit_width=4
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
PY
```

Tune the embedder or quantization width via constructor args:
`TurbovecRetriever(model_name="sentence-transformers/all-mpnet-base-v2", bit_width=8)`.

!!! warning
    If the `turbovec` + `local` extras aren't installed, `.index()` raises
    exactly this (verified on this machine, extras not installed):

    ```
    RuntimeError: turbovec retriever needs the 'turbovec' + 'local' extras:
      uv pip install -e '.[turbovec,local]'
    ```

    Fix: re-run `uv sync --project engine --extra all` and resolve any
    error it reports.

## pi-serini — `PiSeriniRetriever` (extra: `pyserini` + Java 21)

Lucene BM25 via Pyserini/Anserini — the reference lexical retriever from the
pi-serini paper. Use it for Lucene-grade BM25 at scale, or to reproduce the
paper's `k1=25, b=1` tuning for long documents.

The near-identical names are intentional: `pi-serini` is the strategy and
registry key (from the Pi-Serini paper), `pyserini` is the Castorini
library and install extra it runs on.

Setup: synced by the full sync, which needs a Java 21 JDK on `PATH` in
addition to the pip install. If the `pyserini` extra was skipped (or synced
without a JDK present), re-run after installing Java 21:

```bash
uv sync --project engine --extra all
```

```bash
uv run --project engine --extra all python - <<'PY'
from retrieval.document import Document
from retrieval.retrievers import PiSeriniRetriever

docs = [Document("d1", "Routers forward packets between networks and carry data.")]
r = PiSeriniRetriever()  # default k1=0.9, b=0.4; pass k1=25, b=1 for the paper's tuning
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
PY
```

!!! warning
    If the `pyserini` extra or Java 21 is missing, `.index()` raises
    exactly this (verified on this machine, `pyserini` extra not
    installed):

    ```
    RuntimeError: pi-serini retriever needs the 'pyserini' extra and Java 21:
      uv pip install -e '.[pyserini]'   (and install a JDK 21)
    ```

    Fix: re-run `uv sync --project engine --extra all`, install a Java 21
    JDK if needed, and confirm `java -version` reports 21 — a "successful"
    sync with no JDK on `PATH` still fails at index time.

Per-method strengths/weaknesses and the loader-based snippets:
`skills/dense-retrieval-usage/SKILL.md`, `skills/lucene-retrieval-usage/SKILL.md`.

## Next steps

- [Hybrid fusion](hybrid-fusion.md) — combine two retrievers' rankings with RRF.
- [Retrieval strategies compared](../concepts/retrieval-strategies.md)
- [Troubleshooting](../reference/troubleshooting.md)
