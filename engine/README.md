# rag-retrieval

Fully offline contextual RAG retrieval library. Six comparable retrieval
strategies — contextual lexical retrieval (TF-IDF + BM25 fused with
reciprocal-rank fusion), its LLM-enriched `lexical+ctx` variant, turbovec
(dense ANN over quantized embeddings), pi-serini (Lucene BM25), hybrid
(lexical + dense arms fused with RRF), and tree-sitter (AST-boundary chunking
with enclosing-scope context) — with a stdlib-only core and graceful
degradation when optional backends are missing.

This package is the vendored engine of the `agentic-retrieval` Claude Code
plugin. For the full invocation protocol, skills, and setup instructions, see
the parent repository: https://github.com/josix/agentic-retrieval

## Install

From a built wheel or sdist:

```bash
uv pip install rag_retrieval-0.2.0-py3-none-any.whl
```

Directly from GitHub (no local checkout needed), using the `engine/`
subdirectory of the monorepo:

```bash
uv pip install "rag-retrieval @ git+https://github.com/josix/agentic-retrieval.git#subdirectory=engine"
```

With optional extras (`remote`, `local`, `turbovec`, `pyserini`, `treesitter`,
or `all` for every retrieval extra in one install):

```bash
uv pip install "rag-retrieval[all] @ git+https://github.com/josix/agentic-retrieval.git#subdirectory=engine"
```

## Extras

| Extra | Installs | Purpose |
|---|---|---|
| `remote` | `anthropic`, `voyageai`, `cohere`, `httpx`, `requests` | LLM-based contextualization |
| `local` | `sentence-transformers` | Local embeddings for dense retrieval |
| `turbovec` | `turbovec` | TurboQuant-quantized dense ANN retriever |
| `pyserini` | `pyserini` | Lucene BM25 retriever (needs a Java 21 JDK) |
| `treesitter` | `tree-sitter`, `tree-sitter-language-pack` | AST-boundary ("cAST") chunking for code corpora |
| `all` | `rag-retrieval[local,remote,turbovec,pyserini,treesitter]` | Every retrieval extra above, one-shot install of every strategy |
| `dev` | `ruff`, `isort`, `complexipy` | Dev tooling |

The core install (no extras) has zero required dependencies and always works.

## Quick start

```python
from retrieval.document import Document
from retrieval.retrievers import LexicalRetriever

docs = [Document("d1", "Routers forward packets between networks.")]
r = LexicalRetriever()
r.index(docs)
print(r.search("what carries data between networks", top_k=5))
```

## CLI

Installing the package registers a `retrieval` console script
(`[project.scripts]` in `pyproject.toml`):

```bash
retrieval index --root <path-to-project> [--retriever lexical|lexical+ctx] [--force]
retrieval query "<text>" --root <path-to-project> [--retriever lexical|lexical+ctx] [--top-k N] [--json] [--stale-ok]
retrieval stats --root <path-to-project>
retrieval --version
```

- `index` builds a retriever over `load_documents(root)` and persists it as
  a JSON cache (`lexical.json` + `meta.json`) under
  `~/.cache/agentic-retrieval/indexes/<project-key>`, where `<project-key>`
  is a short hash of the root's resolved absolute path — set
  `RETRIEVAL_INDEX_DIR` to relocate the cache base directory (see the
  warning below about pointing it inside the indexed project root). If a
  fresh (non-stale) cache already exists, `index` skips the rebuild and
  prints `index up to date -> <dir>`; pass `--force` to rebuild
  unconditionally.
- `query` loads the cache and searches it, printing one docid per line (best
  match first) or `{"query": ..., "results": [...]}` with `--json`.
- **Stale behavior**: `meta.json` stores a fingerprint (a SHA-256 over every
  discovered file's relative path, size, and mtime). If the root has no
  cache, or the current fingerprint no longer matches (files added, removed,
  or edited), `query` auto-reindexes and re-saves before searching; pass
  `--stale-ok` to search the existing (possibly stale) cache instead.
- `stats` reports the cache's root, doc count, creation time, engine
  version, staleness, and on-disk location without searching or rebuilding.
- `--root` defaults to the `RETRIEVAL_ROOT` environment variable, then the
  current working directory, if omitted.

> **Warning**: if `RETRIEVAL_INDEX_DIR` points at a directory *inside* the
> indexed project root (rather than the default `~/.cache/...` location or
> some other directory outside the root), the cache's `lexical.json`/
> `meta.json` get swept up as documents on the next `index`/`query` run —
> only a directory literally named `.cache` is excluded by default, so this
> can create a feedback loop. Point it outside the project root instead.

Without a console-script install, run the same commands through `uv run` or
`python -m retrieval` (see the parent repository's `SETUP.md`).

See the parent repository's `README.md`, `USAGE.md`, and `SETUP.md` for the
full plugin/skill documentation, the `/retrieval` Claude Code command, and
the `uv`-based build & distribution workflow.
