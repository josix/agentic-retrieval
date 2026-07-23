---
name: code-retrieval-usage
description: This skill should be used when indexing or searching a code/script corpus with the tree-sitter retriever, wanting AST-boundary chunk spans plus an enclosing function/class breadcrumb on hits, or handling the RuntimeError raised when the treesitter extra is not installed.
---

# Code Retrieval Usage (tree-sitter)

AST-boundary ("cAST") chunking for code and script corpora, via
`TreeSitterRetriever` (`retrieval/retrievers.py`), backed by
[tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack)
grammars.

## What it is

`retrieval.ast_chunker.chunk_code` parses a file with tree-sitter and splits
it at AST node boundaries (cAST, arXiv 2506.15655): consecutive sibling nodes
are greedily merged into a chunk while their combined non-whitespace
character count stays under a budget; a node too large to fit alone is
recursed into instead of merged; a leaf node that still doesn't fit is
hard-split by lines. Every chunk keeps the project's 1-based `[start_line,
end_line]` span convention, plus a dotted breadcrumb `context` (e.g.
`"Bar.baz"` for a method `baz` nested in class `Bar`) built by walking the
ancestor scopes the chunker actually recursed into — a whole small file kept
as one chunk gets an empty `context`.

`TreeSitterRetriever` subclasses `LexicalRetriever`: it prefixes each
document's `context` breadcrumb into its ranked text (so a query like
"Bar baz" can match purely via the enclosing-scope name), then ranks with the
same TF-IDF + BM25 + RRF as every other lexical retriever in this plugin.
Tree-sitter is only needed at chunking time
(`retrieval.project_loader.load_ast_chunk_documents`) — the retriever class
itself has zero optional dependencies.

## When to use it (vs lexical / dense / lucene)

**Strengths:**
- Chunk boundaries align to real syntax (function/class/method), not
  paragraph-shaped heuristics — a hit's span is a complete, syntactically
  coherent unit instead of an arbitrary character window.
- Carries a breadcrumb of enclosing scope on every hit, so "which class/method
  is this in" is answered without an extra `Read` + scroll-up.
- Ranking is the same zero-dependency BM25+TF-IDF+RRF as `lexical-retrieval-usage`
  — only the chunking step needs the extra.

**Weaknesses:**
- Needs the `tree-sitter` + `tree-sitter-language-pack` packages (the
  `treesitter` extra) — not zero-dependency.
- Language coverage and scope-node mapping are best-effort
  (`retrieval.ast_chunker.LANGUAGE_BY_SUFFIX` /
  `SCOPE_NODE_TYPES`); an unmapped suffix, or a file whose grammar isn't in
  the installed pack, falls back to the line-based chunker with an empty
  `context` for that file only.
- No semantic similarity and no vocabulary-mismatch bridging — same lexical
  ranking limitations as `lexical-retrieval-usage`.

**Use tree-sitter retrieval when:** the corpus is source code or scripts and
you want spans that respect function/class boundaries plus "what scope is
this in" context on every hit. Compare against `lexical-retrieval-usage` for
prose/mixed corpora or a zero-dependency baseline, `dense-retrieval-usage`
when queries and code are likely to differ in wording, and
`lucene-retrieval-usage` for Lucene-grade BM25 depth at scale. See
`hybrid-retrieval-usage` for the full method-selection decision table.

## Setup

Dependencies (`tree-sitter`, `tree-sitter-language-pack`) come from the
`treesitter` extra, installed by the plugin's one-shot sync:

```bash
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all
```

No global installs. If the sync fails (usually a network/pip-index issue),
resolve the error `uv` reports and re-run the same command.

## How to index and search

Index the invoking project's own files — capture the project root **before**
invoking `uv run`, then feed it to the Python snippet via an env var:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
PROJECT_ROOT="$PROJECT_ROOT" uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all python - <<'PY'
import os

from retrieval.project_loader import load_ast_chunk_documents
from retrieval.retrievers import TreeSitterRetriever

docs = load_ast_chunk_documents(os.environ["PROJECT_ROOT"])

r = TreeSitterRetriever()
r.index(docs)
for hit in r.search_detailed("what carries data between networks", top_k=5):
    scope = f" ({hit.context})" if hit.context else ""
    print(f"{hit.source_path}:{hit.start_line}-{hit.end_line}{scope}")
PY
```

`load_ast_chunk_documents` mirrors `load_chunk_documents` (same file
discovery, same VCS/dependency/build + secret-filename exclusions — see
`lexical-retrieval-usage/references/file-discovery.md`) but chunks each file
at AST boundaries when its suffix maps to a supported language
(`language_for_path`); files with an unmapped suffix, or whose AST chunking
comes back empty, fall back to the line-based `chunk_document` for that file.
Each returned `Document`'s `docid` keeps the `"{path}:{start}-{end}"`
convention, and `search_detailed` hits carry `hit.context` alongside
`hit.source_path`/`hit.start_line`/`hit.end_line`, so a hit turns straight
into `Read(hit.source_path, offset=hit.start_line, limit=hit.end_line -
hit.start_line + 1)` with the enclosing scope already known. Treat that span
as a seed to read and explore from, not the final answer — follow the
references it surfaces outward and re-query with the vocabulary a hit
reveals; if the top spans look noisy, re-query, switch retriever, or raise
`--top-k` (see the `retrieval` skill's Step 3).

Or via the registry: `from retrieval.retrievers import build_retriever;
build_retriever("treesitter")`.

## Graceful degradation

`retrieval.ast_chunker.chunk_code` (called from `load_ast_chunk_documents`)
lazily imports `tree_sitter_language_pack`; if it's missing, it raises:

```
RuntimeError: tree-sitter retriever needs the 'treesitter' extra:
  uv pip install -e '.[treesitter]'
```

`TreeSitterRetriever` itself never raises this — it only needs tree-sitter at
chunking time, not at ranking time — so catch the `RuntimeError` around
`load_ast_chunk_documents` and fall back to `load_chunk_documents` +
`LexicalRetriever` (always available). Fix: re-run
`UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all`
and resolve any error it reports.

A grammar the installed language pack doesn't recognize, or a file whose
parse is too broken to chunk meaningfully, is a **per-file** fallback, not a
hard failure: `chunk_code` returns `[]` for that file and
`load_ast_chunk_documents` chunks it with the line-based `chunk_document`
instead, so one bad file never aborts indexing the rest of the corpus.

## Cross-links

- `lexical-retrieval-usage` — contextual lexical retrieval (zero-dep baseline
  and fallback)
- `dense-retrieval-usage` — turbovec dense ANN retrieval
- `lucene-retrieval-usage` — pi-serini Lucene BM25 retrieval
- `hybrid-retrieval-usage` — fusing dense with lexical, and the
  method-selection decision table
- `docs/how-to/consolidated-query.md` — the default `retrieval query`
  consolidates every available strategy (including `treesitter`) into one
  ranked, deduplicated, explainable list
