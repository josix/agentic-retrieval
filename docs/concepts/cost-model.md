# Cost model

**The default is free.** Nothing in this plugin calls a paid LLM unless you
explicitly wire one in. This page explains the cost model behind LLM
contextualization, for when you do.

## Index-once, search-free

- The LLM only runs **once, at indexing time** — one call per document
  (`ContextualLexicalRetriever`, via `LLMDocumentContextualizer`, model
  `claude-haiku-4-5`) or one call per chunk (`LLMContextualizer`, model
  `claude-opus-4-8` with `effort: "low"`), never per query.
- **Searching is always free** — once the index is built, `.search()` does
  no network calls regardless of which contextualizer built the index.
- Cost still scales with corpus size: more documents/chunks means more LLM
  calls at index time.

## Document vs chunk granularity

The two contextualizers trade cost against precision at different
granularities:

- **Document-level** (`ContextualLexicalRetriever` /
  `LLMDocumentContextualizer`) — one LLM call per *document*, running on
  Haiku (the cheapest tier). This is the cost-sensitive arm: cost scales
  with document count, not chunk count.
- **Chunk-level** (`LLMContextualizer`) — one LLM call per *chunk*, running
  on Opus with `effort: "low"`. This gives finer-grained, chunk-situated
  context (closer to Anthropic's original Contextual Retrieval technique)
  but costs more per document, since a document with many chunks means
  many LLM calls.

Choose document-level when cost matters most and you retrieve whole
documents; choose chunk-level when you retrieve chunks and need the LLM to
see exactly where each chunk sits in the document.

## Prompt caching

The whole document is sent in a `cache_control` (prompt-caching) block, so
every chunk from the same document reuses the cached document prefix
instead of re-sending it — the standard cost mitigation for this
technique, but it does not make chunk-level contextualization free. In the
chunk-level path, the document text is the cached prefix and the per-chunk
instruction is the varying suffix; in the document-level path, the static
system instruction is the cached prefix and the document body (which
differs every call) is the varying, uncached suffix.

## Next steps

- [How to enable LLM contextualization](../how-to/llm-contextualization.md)
- [Contextual retrieval](contextual-retrieval.md)
