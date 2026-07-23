# agentic-retrieval

A Claude Code plugin + skill for comparing six retrieval strategies over
the invoking project's own files (docs/code under the project root). Ships a
vendored, offline-first retrieval engine (`engine/`) with a stdlib-only core
and optional per-strategy extras — no global installs, ever.

All Python dependencies are managed by [`uv`](https://docs.astral.sh/uv/) —
zero global `pip install` anywhere. `uv` is a hard requirement.

```bash
uv sync --project engine --extra all
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT"
uv run --project engine --extra all retrieval query \
  "what carries data between networks" --root "$PROJECT_ROOT" --top-k 5
```

## The retrieval strategies

| Strategy | REGISTRY key | Class | What it is | Extra needed |
|---|---|---|---|---|
| Contextual retrieval | `lexical` | `retrieval.retrievers.LexicalRetriever` | TF-IDF + BM25 fused with reciprocal-rank fusion | none (core) |
| Contextual retrieval + LLM enrichment | `lexical+ctx` | `retrieval.retrievers.ContextualLexicalRetriever` | `LexicalRetriever` over LLM-enriched document text | none (base); `remote` for enrichment |
| turbovec | `turbovec` | `retrieval.retrievers.TurbovecRetriever` | Dense ANN retrieval over embeddings, quantized with TurboQuant | `turbovec` + `local` |
| pi-serini | `pi-serini` | `retrieval.retrievers.PiSeriniRetriever` | Lucene BM25 via Pyserini — the reference lexical retriever from the Pi-Serini paper | `pyserini` (+ Java 21 JDK) |
| hybrid | `hybrid` | `retrieval.retrievers.HybridRetriever` | Lexical + dense arms over the same corpus, fused with reciprocal-rank fusion at search time | `turbovec` + `local` |
| tree-sitter | `treesitter` | `retrieval.retrievers.TreeSitterRetriever` | `LexicalRetriever` over AST-boundary ("cAST") code chunks, carrying an enclosing function/class breadcrumb | none (base); `treesitter` for AST chunking |

The `lexical` (contextual retrieval) strategy is the zero-dependency
baseline: it always runs when searching the invoking project's own files.
`lexical+ctx`, `turbovec`, `pi-serini`, `hybrid`, and `treesitter` are opt-in
comparison retrievers — if their extra isn't installed, calling `.index()`
(or, for `treesitter`, `load_ast_chunk_documents()`) raises a `RuntimeError`
with install instructions rather than crashing silently.

## Where to go next

- New to this plugin? Start with the
  [quickstart tutorial](getting-started/quickstart.md).
- Need to do something specific? See the
  [how-to guides](how-to/install.md) — install, run the CLI, use each
  retriever, fuse rankings, enable LLM contextualization, customize
  indexing, integrate other coding agents, and build/release the engine.
- Want to understand *why* things work the way they do? See
  [concepts](concepts/retrieval-strategies.md) — strategy comparison,
  contextual retrieval, the cost model, and the architecture.
- Looking for exact flags, cache internals, or environment variables? See
  [reference](reference/cli.md) — CLI synopsis, persistence/cache, env
  vars, engine API, troubleshooting, and the generated API pages.
- [Changelog](changelog.md)

## The six plugin skills

As a Claude Code skill, the same flows are driven with `/retrieval`. Six
skills ship with the plugin:

- `skills/retrieval/SKILL.md` — the full invocation protocol (setup, index,
  query) and guardrails.
- `skills/lexical-retrieval-usage/SKILL.md` — contextual lexical retrieval
  (TF-IDF + BM25 + RRF), the zero-dependency baseline.
- `skills/dense-retrieval-usage/SKILL.md` — turbovec dense ANN retrieval.
- `skills/lucene-retrieval-usage/SKILL.md` — pi-serini Lucene BM25 retrieval.
- `skills/code-retrieval-usage/SKILL.md` — tree-sitter AST-boundary chunking
  for code corpora.
- `skills/hybrid-retrieval-usage/SKILL.md` — fusing methods and choosing
  between them.
