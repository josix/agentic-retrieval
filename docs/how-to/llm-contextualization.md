# Enable LLM contextualization

**The default is free.** Nothing in this plugin calls a paid LLM unless you
explicitly wire one in. This page covers how to opt in.

For *why* this helps and the cost model behind it, see
[Contextual retrieval](../concepts/contextual-retrieval.md) and
[Cost model](../concepts/cost-model.md).

## How to enable it

The `remote` extra (`anthropic`, `voyageai`, `cohere`, `httpx`, `requests`)
is synced by the full `uv sync --project engine --extra all` — just export
the API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If the `remote` extra was skipped, sync it first:

```bash
uv sync --project engine --extra all
```

## Three usage paths

### (a) Document-level, the built-in default (Haiku)

Makes real API calls — syntax/imports verified, not executed here:

```python
from retrieval.document import Document
from retrieval.retrievers import ContextualLexicalRetriever

docs = [Document("d1", "Routers forward packets between networks and carry data.")]
r = ContextualLexicalRetriever()  # lazily builds LLMDocumentContextualizer (claude-haiku-4-5)
r.index(docs)                     # <-- one Haiku call per document happens here
print(r.search("what carries data between networks", top_k=5))
```

### (b) Chunk-level, explicit injection (Opus, `effort: "low"`)

Makes real API calls — syntax/imports verified, not executed here:

```python
from retrieval.index import ContextualRetriever
from retrieval.llm_contextualizer import LLMContextualizer, contextualize_llm
from retrieval.project_loader import load_chunks

chunks = load_chunks(project_root)

retriever = ContextualRetriever()
# Either the module-level one-liner...
retriever.build(chunks, contextualizer=contextualize_llm)
# ...or an explicit instance (e.g. to pick a non-default model):
retriever.build(chunks, contextualizer=LLMContextualizer(model="claude-opus-4-8").generate)
```

### (c) Keep it free — inject the heuristic instead of an LLM

This is runnable as-is, no API key needed:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJECT_ROOT="$PROJECT_ROOT" uv run --project engine --extra all python - <<'PY'
import os

from retrieval.contextualizer import contextualize
from retrieval.index import ContextualRetriever
from retrieval.project_loader import load_chunks

chunks = load_chunks(os.environ["PROJECT_ROOT"])

retriever = ContextualRetriever()
retriever.build(chunks, contextualizer=contextualize)  # heuristic, no API key needed
results = retriever.search("what carries data between networks", top_k=3)
for r in results:
    print(r.fused_rank, r.chunk.doc_title, r.chunk.heading)
PY
```

`ContextualRetriever.build(chunks, use_context, contextualizer)` accepts any
`(chunk, doc_chunks) -> str` callable in the `contextualizer=` slot —
passing the heuristic here is how you verify the wiring (and get the recall
benefit of contextualization) before ever paying for an LLM call.

## `RAG_PROVIDER` stubs

`retrieval/providers.py::get_contextualizer()` reads `RAG_PROVIDER` (default
`"heuristic"`, fully offline and what you get by default). The other three
values — `"ollama"`, `"remote"`, `"sentence_transformers"` — are currently
**stubs** that always raise `RuntimeError` on use; they're wiring points for
future providers, not working integrations.

!!! warning
    Don't set `RAG_PROVIDER` expecting real contextualization from
    `"ollama"`, `"remote"`, or `"sentence_transformers"` — use the heuristic
    default, or `retrieval.llm_contextualizer` directly (paths (a)/(b)
    above) for real LLM contextualization.

## Cost-control checklist

- Default is heuristic — you get free contextualization
  (`ContextualRetriever.build(chunks, use_context=True)`, or any
  `contextualizer=` slot left at its default) unless you deliberately inject
  an LLM contextualizer.
- The LLM only runs when you wire it in — `ContextualLexicalRetriever()`
  with no args, or `contextualizer=contextualize_llm` /
  `LLMContextualizer(...).generate`.
- Index once, search free — the API cost is entirely at `.index()`/`.build()`
  time; every subsequent `.search()` call is local computation.
- If the `remote` group or `ANTHROPIC_API_KEY` is missing, you get a
  `RuntimeError` guard (`"LLM contextualizer needs the 'anthropic' package +
  ANTHROPIC_API_KEY"`) instead of a silent charge or crash — catch it and
  fall back to the heuristic or plain `LexicalRetriever`.

## Next steps

- [Contextual retrieval concepts](../concepts/contextual-retrieval.md)
- [Cost model](../concepts/cost-model.md)
- [Troubleshooting](../reference/troubleshooting.md)
