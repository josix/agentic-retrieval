# agentic-retrieval

A Claude Code plugin + skill for comparing six retrieval strategies over
the invoking project's own files (docs/code under the project root). Ships a
vendored, offline-first retrieval engine (`engine/`) with a stdlib-only core
and optional per-strategy extras — no global installs, ever.

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

## Install

All Python dependencies are managed by [`uv`](https://docs.astral.sh/uv/) —
zero global `pip install` anywhere. `uv` is a hard requirement.

```bash
uv sync --project engine --extra all
```

## Usage

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT"
uv run --project engine --extra all retrieval query \
  "what carries data between networks" --root "$PROJECT_ROOT" --top-k 5
```

`index` with no `--retriever` builds five of the six strategy caches by
default — `lexical`, `turbovec`, `pi-serini`, `hybrid`, and `treesitter`
(`lexical+ctx` is opt-in only, since it shares the `lexical` cache slot and
costs LLM tokens).
`lexical` always builds; the others build whenever their extras are present —
a missing extra is skipped with a note, never a hard failure.
`query` then picks `--retriever` per question (default `lexical`) — the
coding agent chooses which cached strategy fits each query.

As a Claude Code skill, the same flows are driven with `/retrieval`.

## Full documentation

The full Diataxis-organized documentation site lives at
<https://josix.github.io/agentic-retrieval/> (also browsable directly
under [`docs/`](docs/) in this repo) — tutorials, how-to guides, concepts,
and reference material, including the full CLI synopsis, persistence/cache
internals, environment variables, and the generated API pages.

For per-method detail (strengths/weaknesses, setup, concrete snippets,
graceful degradation) and for combining methods, see the six plugin
skills:

- `skills/retrieval/SKILL.md`
- `skills/lexical-retrieval-usage/SKILL.md`
- `skills/dense-retrieval-usage/SKILL.md`
- `skills/lucene-retrieval-usage/SKILL.md`
- `skills/code-retrieval-usage/SKILL.md`
- `skills/hybrid-retrieval-usage/SKILL.md`

## Layout

```
.claude-plugin/plugin.json            plugin manifest (name: agentic-retrieval)
.claude-plugin/marketplace.json       single-plugin marketplace entry
commands/retrieval.md                 thin dispatcher -> skill
skills/retrieval/SKILL.md             full invocation protocol
skills/lexical-retrieval-usage/       knowledge skill: contextual lexical retrieval
skills/dense-retrieval-usage/         knowledge skill: turbovec dense ANN
skills/lucene-retrieval-usage/        knowledge skill: pi-serini Lucene BM25
skills/code-retrieval-usage/          knowledge skill: tree-sitter AST-boundary chunking
skills/hybrid-retrieval-usage/        knowledge skill: fusion + method selection
engine/                               vendored offline-first retrieval engine (agentic-retrieval package)
  pyproject.toml
  uv.lock
  retrieval/  tests/
    retrieval/retrievers.py             LexicalRetriever, ContextualLexicalRetriever, TurbovecRetriever, PiSeriniRetriever, HybridRetriever, TreeSitterRetriever, REGISTRY, build_retriever
    retrieval/document.py               Document record shared by the retrievers
    retrieval/ast_chunker.py            AST-boundary ("cAST") chunking via tree-sitter
    retrieval/project_loader.py         discover_files/load_documents/load_chunks/load_ast_chunk_documents over a project root
    tests/fixtures/                     engineered fixtures for the Routing-chunk contextualization test
docs/                                  Diataxis-organized MkDocs site (tutorials, how-to, concepts, reference)
```

## Build & distribute

The engine (`engine/`) is the installable `agentic-retrieval` package. See
[`docs/how-to/build-and-release.md`](docs/how-to/build-and-release.md) for
building a wheel/sdist with `uv build`, installing it (locally, or straight
from GitHub via a `git+...#subdirectory=engine` URL), the release workflow,
and the manual docs deploy command.

All active engine code lives under `engine/`, vendored from
`contextual-retrieval-exp`. (A legacy `reference` symlink to a sibling
directory used to sit at the repo root; it has been removed.)
