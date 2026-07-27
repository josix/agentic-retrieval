# `retrieval.consolidation`

Merges per-retriever `SearchHit` rankings into a single deduplicated,
explainable ranking — a starting ranking a following conversation/agent
verifies and explores from. Confidence and score reflect cross-retriever
agreement on query-text match, not currency or authority — a "high" hit can
still point at legacy or deprecated code, and consumers must verify each
span against the live source before relying on it.

::: retrieval.consolidation
