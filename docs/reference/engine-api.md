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
    def search_detailed(self, query: str, top_k: int) -> List[SearchHit]: ...
```

- `name` — a human-readable label for results tables.
- `index(documents)` — build the retriever over a list of `Document`.
  Optional-dependency retrievers may raise `RuntimeError` here if their
  extra isn't installed. Each retriever also builds an internal
  `self._units: List[dict]` (`{docid, source_path, start_line, end_line}`)
  straight from the indexed Documents' span metadata.
- `search_detailed(query, top_k)` — return up to `top_k` ranked
  `SearchHit`s, best match first, each carrying the chunk's
  `docid`/`source_path`/`start_line`/`end_line`/`rank`. Raises
  `RuntimeError("call index() before search()")` if called before a
  successful `index()`.
- `search(query, top_k)` — a thin projection: `[h.docid for h in
  search_detailed(query, top_k)]`. Kept for backward-compatible callers
  that only need docids.

## `SearchHit` dataclass

```python
@dataclass
class SearchHit:
    docid: str
    source_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    rank: int
```

The result type `search_detailed` returns. `source_path`/`start_line`/
`end_line` are resolved from the indexed `Document`'s own span metadata —
never by parsing `docid`. Turn a hit into exact file content with
`Read(hit.source_path, offset=hit.start_line, limit=hit.end_line -
hit.start_line + 1)`.

## `REGISTRY` keys

`retrieval.retrievers.REGISTRY` maps CLI/config names to retriever classes:

| Key | Class | Always available? |
|---|---|---|
| `lexical` | `LexicalRetriever` | Yes — zero dependencies |
| `lexical+ctx` | `ContextualLexicalRetriever` | Yes for base; needs `ANTHROPIC_API_KEY` + `remote` extra for LLM enrichment |
| `turbovec` | `TurbovecRetriever` | No — `RuntimeError` if `turbovec`/`local` extras missing |
| `pi-serini` | `PiSeriniRetriever` | No — `RuntimeError` if `pyserini` extra or Java 21 missing |
| `hybrid` | `HybridRetriever` | No — lexical + dense arms fused with RRF; `RuntimeError` if `turbovec`/`local` extras missing |
| `treesitter` | `TreeSitterRetriever` | Yes for ranking (zero dependencies); `load_ast_chunk_documents()` raises `RuntimeError` if the `treesitter` extra is missing |

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
    source_path: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
```

Pure stdlib, zero dependencies — the shared record every retriever's
`index()`/`search()` operates on. `source_path`/`start_line`/`end_line`
default to `""`/`None`/`None` so whole-file callers are unaffected; a
chunk-granularity `Document` (from `load_chunk_documents`) sets all three,
and that's what every retriever's `search_detailed` resolves a
`SearchHit`'s span from. Kept separate from `retrieval.index.Chunk` (used
by the chunk-level `ContextualRetriever` and the LLM contextualizers).

## `load_chunk_documents`

```python
def load_chunk_documents(root, **kw) -> List[Document]
```

The production loader (`retrieval.project_loader`): discovers files under
`root` (same rules as `load_documents`/`discover_files`), chunks each one
(`retrieval.chunker.chunk_document`), and returns one chunk-granularity
`Document` per span. `docid` is `"{path}:{start}-{end}"` (the file's
relative POSIX path plus its 1-based `[start_line, end_line]` span);
`source_path`/`start_line`/`end_line` are set from the chunk's span. This
is what `retrieval index`/`query` (and every production retriever's
`index()`) build over — `load_documents`/`load_chunks` remain unchanged for
whole-file and raw-`Chunk` use cases respectively.

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
