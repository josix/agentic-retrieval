---
name: retrieval
description: Compare five retrieval strategies — contextual retrieval (TF-IDF/BM25/RRF fusion), its LLM-enriched lexical+ctx variant, turbovec (dense ANN), pi-serini (Lucene BM25), and hybrid (lexical + dense fused with RRF) — with a stdlib-only offline core and graceful degradation when optional backends are missing. Use when the user wants to compare retrieval strategies on their project, search project files, or set up the retrieval engine.
trigger: /retrieval
---

# /retrieval

Compare five retrieval strategies on the invoking project's own files
(docs/code under the project root):

1. **Contextual retrieval** (`lexical`) — TF-IDF + BM25 fused with
   reciprocal-rank fusion. Fully offline, zero required dependencies.
2. **Contextual retrieval + LLM enrichment** (`lexical+ctx`) — the same
   lexical retriever over text enriched by an LLM (or heuristic)
   contextualizer that situates each chunk or document before indexing.
   Opt-in (costs LLM tokens); needs the `remote` extra + `ANTHROPIC_API_KEY`
   for LLM enrichment.
3. **turbovec** — dense ANN retrieval over embeddings, quantized with
   TurboQuant. Needs `sentence-transformers` + `turbovec`, installed by setup.
4. **pi-serini** — Lucene BM25 via Pyserini, the reference lexical retriever
   from the Pi-Serini paper. Needs `pyserini` (installed by setup) + a Java 21
   JDK.
5. **hybrid** — lexical + dense arms over the same corpus, fused with RRF at
   search time. Needs the same extras as turbovec.

The default `lexical` retriever runs fully offline with zero required
dependencies. Every optional strategy degrades gracefully — a missing
backend is skipped with a note, never a hard failure.

## Usage

```
/retrieval setup
/retrieval index
/retrieval query "your query here"
```

`index` with no `--retriever` builds every strategy's cache in one pass
(lexical always; turbovec/pi-serini/hybrid when their extras are present,
else skipped with a note — never a hard failure). `query` then lets the
coding agent pick `--retriever` per question — see "Agent routing" below.

## What You Must Do When Invoked

Follow these steps in order. Do not skip steps.

### Step 1 — Ensure uv is available, then sync the environment

`uv` is the hard requirement for this path. Guard for it, then `uv sync`
the engine's environment with every optional extra — `uv sync` is
idempotent, so it is always safe to re-run:

```bash
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

Standalone (outside the plugin, running from the repo root):

```bash
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
uv sync --project engine --extra all
```

Core (no extras) must always succeed. If the full `--extra all` sync fails
it is almost always the `pyserini` extra (needs a Java 21 JDK on `PATH`; all
extras also need network access). Sync core only to confirm the base engine
works, then re-run the full sync once the prerequisite is in place:

```bash
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}"
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine"             # core only
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all # retry full install
```

Report which extras synced and which failed **verbatim** from `uv`'s own
output — do not paraphrase which groups succeeded.

### Step 2 — Search the project's own files (zero deps, always works)

Capture the project root **before** invoking `uv run` — the skill's working
directory does not change, but `--root`/`RETRIEVAL_ROOT` must point at the
invoking project, not the plugin's `engine/`. Use the `retrieval` console
script (installed by `setup`'s `uv sync`) to build a persisted, on-disk index
once with `index`, then search it (as many times as you like) with `query`:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all \
  retrieval index --root "$PROJECT_ROOT"

UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all \
  retrieval query "your query here" --root "$PROJECT_ROOT" --top-k 5
```

`index` with no `--retriever` (the default, equivalent to `--retriever all`)
builds every default strategy — `lexical` (TF-IDF + BM25 fused with RRF, the
zero-dependency baseline), `turbovec` (dense ANN; needs the turbovec + local
extras), `pi-serini` (Lucene BM25; needs the pyserini extra + Java 21), and
`hybrid` (lexical + dense fused with RRF; needs the turbovec extras) — each
persisted as its own JSON cache slot under
`~/.cache/agentic-retrieval/indexes`, keyed by the project's resolved
path (override the cache location with `RETRIEVAL_INDEX_DIR`), so they
coexist and never invalidate each other. One line per strategy is printed
(`lexical: indexed N docs`, etc.); any strategy whose extras are missing
prints `<name>: skipped (<reason>)` and the run still exits 0, since the
always-available `lexical` strategy succeeding is what matters for the
default run. Pass `--retriever lexical|turbovec|pi-serini|hybrid` to build
just one strategy instead — in that single-strategy form a missing extra
**hard-fails** with a guidance `RuntimeError` (exit 1) naming the install
command, instead of being skipped; fall back to `--retriever lexical` in
that case. `query` loads the cache for whichever `--retriever` the agent
picked and searches it, printing one docid per line (best match first, no
scores) — pass `--json` for `{"query": ..., "results": [...]}` instead. If
that strategy's cache is missing, or the project's files have changed since
it was built (detected by a content fingerprint, no manual invalidation
needed), `query` auto-reindexes just that strategy before searching (also
hard-failing on a missing extra, exit 1 — `query` never silently falls
back); pass `--stale-ok` to search the stale cache anyway instead.
`retrieval stats --root "$PROJECT_ROOT"` reports every built cache slot's
location, doc count, creation time, and staleness without searching.

### Agent routing — which retriever to query per question

After a default `index` run populates every available cache slot, choose
`--retriever` per query based on the question's shape:

| Query characteristic | Query with |
| --- | --- |
| Exact keywords / identifiers / code tokens; zero-dep default | `--retriever lexical` |
| Paraphrase / synonyms / wording differs from documents | `--retriever turbovec` |
| Lucene-grade BM25 depth/scale needed | `--retriever pi-serini` |
| Uncertain — vocabulary mismatch vs genuine irrelevance | `--retriever hybrid` |
| Chosen backend skipped/errored | fall back to `--retriever lexical` |

See `hybrid-retrieval-usage` for the full decision walkthrough.

For LLM or heuristic contextualization before indexing (closing
vocabulary-mismatch gaps), see `lexical-retrieval-usage` /
`hybrid-retrieval-usage`.

**Failure mode** — if `load_documents` finds no indexable files under the
project root, `index` persists an empty (zero-doc) cache rather than raising
or exiting, and `query` then returns no results. Check that
`--root`/`RETRIEVAL_ROOT`/`PROJECT_ROOT` points at the intended project
directory, and remember the default exclusions (`.git`, `.venv`,
`node_modules`, etc. — see
`retrieval/project_loader.py::DEFAULT_EXCLUDE_DIRS`) may have skipped
everything if it points at an already-excluded or empty directory.

### Advanced: direct engine API

The CLI above wraps the same `load_documents` -> `index` -> `search` flow
this heredoc form runs in-memory, per invocation, with no persisted cache —
useful for one-off scripting or when you need the `Retriever` objects
directly rather than the CLI's docid/JSON output:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
RETRIEVAL_ROOT="$PROJECT_ROOT" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import load_documents
from retrieval.retrievers import LexicalRetriever

docs = load_documents(os.environ["RETRIEVAL_ROOT"])
r = LexicalRetriever()
r.index(docs)
for docid in r.search("your query here", top_k=5):
    print(docid)
PY
```

Standalone (outside the plugin, running from the repo root):

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
RETRIEVAL_ROOT="$PROJECT_ROOT" uv run --project engine --extra all python - <<'PY'
import os

from retrieval.project_loader import load_documents
from retrieval.retrievers import LexicalRetriever

docs = load_documents(os.environ["RETRIEVAL_ROOT"])
r = LexicalRetriever()
r.index(docs)
for docid in r.search("your query here", top_k=5):
    print(docid)
PY
```

## Engine API — retrievers

The five retrieval methods are implemented as classes in
`engine/retrieval/retrievers.py`, all built over the shared
`engine/retrieval/document.py::Document` record
(`docid: str`, `text: str`, `url: str = ""`). Select a class either directly
or via `retrieval.retrievers.REGISTRY` / `build_retriever(name)`.

| Method | Class | REGISTRY key | Optional extra | Always available? |
| --- | --- | --- | --- | --- |
| Contextual lexical retrieval (TF-IDF + BM25 + RRF) | `LexicalRetriever` | `lexical` | none (core) | Yes |
| Contextual lexical + LLM-enriched text | `ContextualLexicalRetriever` | `lexical+ctx` | `[remote]` for enrichment | Yes for base; needs `ANTHROPIC_API_KEY` for enrichment |
| turbovec (dense ANN) | `TurbovecRetriever` | `turbovec` | `[turbovec,local]` | No — `RuntimeError` if uninstalled |
| pi-serini (Lucene BM25) | `PiSeriniRetriever` | `pi-serini` | `[pyserini]` | No — `RuntimeError` if uninstalled or Java missing |
| hybrid (lexical + dense RRF fusion) | `HybridRetriever` | `hybrid` | `[turbovec,local]` | No — `RuntimeError` if uninstalled |

```python
from retrieval.retrievers import build_retriever
from retrieval.document import Document

docs = [Document("d1", "Routers forward packets between networks.")]
r = build_retriever("lexical")
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
```

## Graceful degradation

- The core `uv sync` (no extras) must succeed — everything downstream
  depends on it.
- If the full `--extra all` sync fails, sync core only to confirm the base
  engine works, then re-run the full sync once the prerequisite (usually a
  Java 21 JDK for `pyserini`, or network access) is in place — see Step 1's
  core-only fallback.
- `TurbovecRetriever.index()` and `PiSeriniRetriever.index()` raise a
  `RuntimeError` with install instructions when their optional dependency is
  missing — catch this and fall back to `LexicalRetriever` rather than
  aborting.
- The `lexical` retriever always runs — it is the zero-dependency baseline
  every other strategy is compared against.
- `retrieval index` (default, all-strategy mode) degrades gracefully: a
  missing backend prints `<name>: skipped (<reason>)` and the run still
  exits 0, as long as `lexical` itself built successfully.
- `retrieval index --retriever <name>` (single-strategy) and
  `retrieval query --retriever <name>` do **not** degrade — a missing
  backend hard-fails with a guidance `RuntimeError` and exit 1, with no
  silent fallback. If you want graceful behavior at query time, catch the
  failure yourself and retry with `--retriever lexical`.

## Guardrails

- **Cost warnings**: LLM-enriched indexing (`lexical+ctx`, or
  `retrieval.llm_contextualizer`) calls a network LLM and costs tokens — warn
  the user and prefer explicit opt-in before running it without being asked.
- **Anthropic-native default**: when a strategy needs an LLM (contextualizer),
  default to Anthropic's API (`ANTHROPIC_API_KEY`) unless the user asks for a
  different provider.
- **No global installs, no bare `python`/`pip`**: every dependency lives in
  the `uv`-managed environment for `engine/` (at
  `$UV_PROJECT_ENVIRONMENT`, default `$HOME/.cache/agentic-retrieval/uv-venv`)
  — always invoke the engine via `uv run --project <path-to-engine>`, never a
  bare `python`/`python3`/`pip install` call, which would bypass that
  isolation and resolve to a different (likely dependency-less) interpreter.

## Related knowledge skills

For per-method detail (strengths/weaknesses, setup, concrete index/search
snippets, graceful degradation) and for combining methods, see:

- `lexical-retrieval-usage` — contextual lexical retrieval (TF-IDF + BM25 + RRF)
- `dense-retrieval-usage` — turbovec dense ANN retrieval
- `lucene-retrieval-usage` — pi-serini Lucene BM25 retrieval
- `hybrid-retrieval-usage` — fusing methods and choosing between them
