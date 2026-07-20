# Provider selection (experimental)

Referenced from `../SKILL.md` ("Provider selection").

`retrieval/providers.py::get_contextualizer()` picks a contextualizer/embedder
based on the `RAG_PROVIDER` env var, defaulting to `"heuristic"`
(`HeuristicContextualizer`, fully offline). The other three choices —
`"ollama"`, `"remote"`, and `"sentence_transformers"` — are currently **stubs**
that always raise `RuntimeError` on use (they exist as the wiring point for
future providers, not working integrations yet). Do not rely on
`RAG_PROVIDER=ollama|remote|sentence_transformers` for real contextualization
today; use the heuristic default, or `retrieval.llm_contextualizer` directly
for real LLM contextualization (see `../SKILL.md`'s "Contextualization boost"
section, and `custom-contextualizer.md` for the injection wiring).
