# How to index chunk-level (feeds `ContextualRetriever`)

Referenced from `../SKILL.md` ("How to index chunk-level").

`load_chunks` discovers files, chunks each one (`retrieval/chunker.py`), and
returns a flat `List[Chunk]` — the input `ContextualRetriever`
(`retrieval/index.py`) expects, as opposed to the whole-document `Document`
list `LexicalRetriever` / `TurbovecRetriever` / `PiSeriniRetriever` expect:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.index import ContextualRetriever
from retrieval.project_loader import load_chunks

chunks = load_chunks(os.environ["PROJECT_ROOT"])  # same discovery kwargs as load_documents
print("chunks:", len(chunks))

retriever = ContextualRetriever()
retriever.build(chunks, use_context=True)  # offline heuristic contextualizer
results = retriever.search("what carries data between networks", top_k=5)
for r in results:
    print(r.fused_rank, r.chunk.doc_title, r.chunk.heading)
PY
```

`ContextualRetriever.build` also accepts an injected `contextualizer`
callable (e.g. the LLM-based one) instead of the heuristic — see
`hybrid-retrieval-usage` for that wiring.
