# Contextual retrieval: heuristic vs LLM

Vocabulary-mismatch chunks — where the query's words never literally appear
in the relevant chunk — are exactly the failure mode plain lexical
retrieval can't fix on its own. Contextualization closes that gap by
prepending a short piece of context to each chunk (or document) before
indexing, giving the sparse retriever (TF-IDF/BM25) extra surface
vocabulary to match against.

This plugin ships two contextualizers with the same
`(chunk, doc_chunks) -> str` contract, so either can be dropped into
`ContextualRetriever.build(chunks, contextualizer=...)`.

## The free default: the heuristic contextualizer

`retrieval/contextualizer.py::contextualize` is deterministic, offline, and
free — it prepends a structural breadcrumb (`[doc_title > heading]` + the
heading's words + the previous chunk's first sentence) to each chunk before
indexing. This is what `ContextualRetriever.build(chunks, use_context=True)`
uses when no explicit `contextualizer` is given, and it's injectable
anywhere the engine accepts a `contextualizer` callable. Zero API calls,
zero tokens, zero cost — this is the path to stay on by default.

The breadcrumb format degrades gracefully: when a chunk has no heading, the
breadcrumb drops to `[doc_title]`; when there's no preceding chunk in the
same document, the previous-sentence line is simply omitted.

## Why an LLM helps

Per Anthropic's Contextual Retrieval reference study, an LLM-written context
per chunk closes the vocabulary-mismatch gap and recovers recall on those
ambiguous chunks better than the free heuristic can, because the LLM
actually reads the whole document and writes context in natural language
rather than following a fixed breadcrumb template. Where the heuristic only
knows a chunk's heading and its immediate neighbor, the LLM sees the entire
document and can name topics, entities, and relationships that never appear
verbatim near the chunk.

`retrieval/llm_contextualizer.py::LLMContextualizer` implements this at the
chunk level: for each chunk, an LLM is shown the whole document and the
chunk, and writes a short context (one or two sentences) that situates the
chunk within the document. The document text is sent once per document in
a `cache_control` block, so every chunk from the same document reuses that
cached prefix (see [cost model](cost-model.md)).

`LLMDocumentContextualizer` is the document-granularity analog, used by
`ContextualLexicalRetriever` — since that retriever ranks whole documents
rather than chunks, there's no parent document to situate a chunk within,
so it asks the LLM for a short context (topics + key entities) for the
*whole document* instead, and runs on Haiku (the cheapest tier) since it's
the cost-sensitive arm — one call per document rather than one per chunk.

## Choosing between them

Use the heuristic by default — it's free and injectable exactly like the
LLM contextualizer, so you can verify the wiring (and get part of the
recall benefit) before ever paying for an LLM call. Reach for the LLM
contextualizer specifically when sparse retrieval keeps missing documents
or chunks for lack of shared vocabulary, and standing up a dense/Lucene
comparison retriever isn't worth it for your corpus.

## Next steps

- [How to enable LLM contextualization](../how-to/llm-contextualization.md)
- [Cost model](cost-model.md)
- [Reference: contextualizer API](../reference/api/contextualizers.md)
