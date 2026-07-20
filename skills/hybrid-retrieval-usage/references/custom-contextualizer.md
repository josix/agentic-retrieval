# Wiring a custom contextualizer into `ContextualRetriever`

Referenced from `../SKILL.md` ("Wiring a custom contextualizer into
`ContextualRetriever`").

`ContextualRetriever.build(chunks, use_context=True, contextualizer=...)`
(`retrieval/index.py`) accepts an injected `(chunk, doc_chunks) -> str`
callable that overrides the built-in heuristic — this is how the chunk-level
retriever is wired up to real LLM contextualization instead of the offline
heuristic.

**Requires the `remote` group installed by setup + `ANTHROPIC_API_KEY`; costs
one LLM call per chunk — warn the user and prefer opt-in before running
it.** Syntax only below, not executed here:

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

The same `contextualizer=` slot accepts any callable with that signature —
including a plain heuristic one, which is safe to run without an API key and
lets you verify the wiring before paying for LLM calls:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
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
