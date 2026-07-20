# Engine API overview

Hand-written overview of the `retrieval` package's public surface. For full
generated signatures/docstrings, see the API pages linked below.

## The `Retriever` protocol contract

Every retriever implements the same tiny contract (`retrieval.retrievers.Retriever`,
a `runtime_checkable` `Protocol`):

```python
class Retriever(Protocol):
    name: str

    def index(self, documents: List[Document]) -> None: ...
    def search(self, query: str, top_k: int) -> List[str]: ...
```

- `name` — a human-readable label for results tables.
- `index(documents)` — build the retriever over a list of `Document`.
  Optional-dependency retrievers may raise `RuntimeError` here if their
  extra isn't installed.
- `search(query, top_k)` — return up to `top_k` ranked docids, best match
  first. Raises `RuntimeError("call index() before search()")` if called
  before a successful `index()`.

## `REGISTRY` keys

`retrieval.retrievers.REGISTRY` maps CLI/config names to retriever classes:

| Key | Class | Always available? |
|---|---|---|
| `lexical` | `LexicalRetriever` | Yes — zero dependencies |
| `lexical+ctx` | `ContextualLexicalRetriever` | Yes for base; needs `ANTHROPIC_API_KEY` + `remote` extra for LLM enrichment |
| `turbovec` | `TurbovecRetriever` | No — `RuntimeError` if `turbovec`/`local` extras missing |
| `pi-serini` | `PiSeriniRetriever` | No — `RuntimeError` if `pyserini` extra or Java 21 missing |
| `hybrid` | `HybridRetriever` | No — lexical + dense arms fused with RRF; `RuntimeError` if `turbovec`/`local` extras missing |

## `build_retriever`

```python
def build_retriever(name: str) -> Retriever
```

Looks up `name` in `REGISTRY` and returns a freshly constructed instance;
raises `ValueError` for an unknown name.

```python
from retrieval.retrievers import build_retriever

r = build_retriever("lexical")
```

## `Document` dataclass

```python
@dataclass
class Document:
    docid: str
    text: str
    url: str = ""
```

Pure stdlib, zero dependencies — the shared record every retriever's
`index()`/`search()` operates on at document granularity. Kept separate
from `retrieval.index.Chunk` (used by the chunk-level `ContextualRetriever`
and the LLM contextualizers), since `retrieval.retrievers` operates at
document granularity while `retrieval.index`/`retrieval.chunker` operate at
chunk granularity.

## Generated API pages

- [`retrieval.retrievers`](api/retrievers.md)
- [`retrieval.document`](api/document.md)
- [`retrieval.project_loader`](api/project_loader.md)
- [`retrieval.fusion`](api/fusion.md)
- [Contextualizers](api/contextualizers.md) (`retrieval.contextualizer` +
  `retrieval.llm_contextualizer`)

## Next steps

- [Retrieval strategies compared](../concepts/retrieval-strategies.md)
- [Architecture](../concepts/architecture.md)
