"""LLM-based contextualizer — Anthropic's original Contextual Retrieval.

Where the heuristic contextualizer (``retrieval/contextualizer.py``) prepends a
deterministic breadcrumb, this implements the technique from Anthropic's
"Contextual Retrieval" post: for each chunk, an LLM is shown the *whole
document* and the chunk, and writes a short context that situates the chunk
within the document.  That context is prepended before indexing, giving the
sparse retriever vocabulary the chunk body lacks.

This is an **opt-in, online** comparison arm — it needs the ``anthropic``
package and ``ANTHROPIC_API_KEY``.  The import is lazy (inside methods), so the
default offline pipeline and the no-network-imports test are unaffected.

The whole document is sent in a ``cache_control`` block, so every chunk from the
same document reuses the cached document prefix (the standard cost optimization
for this technique).
"""

from typing import List, Optional

from retrieval.chunker import Chunk

_MODEL = "claude-opus-4-8"

# The document-level (``lexical+ctx``) arm is the cost-sensitive one — one LLM
# call per document — so it runs on Haiku.  Note: ``output_config.effort`` is
# NOT supported on Haiku 4.5 (it 400s), so the document contextualizer omits it.
_DOC_MODEL = "claude-haiku-4-5"

_SITUATE_INSTRUCTION = (
    "Here is a chunk we want to situate within the whole document above:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context (one or two sentences) to situate this "
    "chunk within the overall document for the purposes of improving search "
    "retrieval of the chunk. Answer ONLY with the succinct context and nothing "
    "else."
)

# Static, document-independent instruction.  It's the only part of the
# document-level prompt shared across every call, so it lives in a cached
# ``system`` prefix (the document body — which differs every call — is the
# varying suffix and is not cached).
_DOC_SYSTEM = (
    "Write a short, succinct context (two or three sentences) capturing what "
    "the document the user provides is about — its main topics, key entities, "
    "and any names or terms a searcher might use to find it. The goal is to "
    "improve search retrieval of the document. Answer ONLY with the succinct "
    "context and nothing else."
)


def _new_anthropic_client():
    """Lazily construct an Anthropic client (keeps the core offline)."""
    try:
        import anthropic  # noqa: F401  # lazy: keeps the core offline
    except ImportError as exc:  # pragma: no cover - guidance path
        raise RuntimeError(
            "LLM contextualizer needs the 'anthropic' package + "
            "ANTHROPIC_API_KEY:\n  uv pip install -e '.[remote]'"
        ) from exc
    return anthropic.Anthropic()


class LLMContextualizer:
    """Situate each chunk within its document via an LLM (Contextual Retrieval).

    Exposes the same ``generate(chunk, all_chunks) -> str`` contract as the
    contextualizers in ``retrieval/providers.py``.
    """

    def __init__(self, model: str = _MODEL) -> None:
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = _new_anthropic_client()
        return self._client

    @staticmethod
    def _document_text(all_chunks_in_doc: List[Chunk]) -> str:
        """Reconstruct the document by joining its chunks in position order."""
        ordered = sorted(all_chunks_in_doc, key=lambda c: c.position)
        return "\n\n".join(c.text for c in ordered)

    def _situate(self, document: str, chunk_text: str) -> str:
        client = self._ensure_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=200,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{document}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": _SITUATE_INSTRUCTION.format(chunk=chunk_text),
                        },
                    ],
                }
            ],
        )
        return next((b.text for b in response.content if b.type == "text"), "").strip()

    def generate(self, chunk: Chunk, all_chunks_in_doc: List[Chunk]) -> str:
        """Return ``<llm context> <chunk text>`` for indexing."""
        document = self._document_text(all_chunks_in_doc)
        context = self._situate(document, chunk.text)
        return f"{context} {chunk.text}".strip()


class LLMDocumentContextualizer:
    """Document-level context enrichment (document-granularity analog).

    Used by ``ContextualLexicalRetriever`` (``retrieval/retrievers.py``), which
    retrieves whole documents rather than chunks, so there is no parent
    document to situate a chunk within.  The document-level adaptation of
    Contextual Retrieval is to ask the LLM for a short context (topics + key
    entities) for the *whole document* and prepend it before indexing — giving
    the sparse retriever extra surface vocabulary.

    Runs on Haiku (the cheapest tier) since this is the cost arm — one call per
    document.  The static instruction sits in a ``cache_control`` ``system``
    block so the shared prefix is reused across documents; the document body is
    the varying suffix and isn't cached.

    Exposes ``generate(document_text) -> str`` returning the context string
    (the caller prepends it to the document).
    """

    def __init__(self, model: str = _DOC_MODEL) -> None:
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = _new_anthropic_client()
        return self._client

    def generate(self, document_text: str) -> str:
        """Return a short LLM-written context describing the document."""
        client = self._ensure_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=256,
            system=[
                {
                    "type": "text",
                    "text": _DOC_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"<document>\n{document_text}\n</document>",
                }
            ],
        )
        return next((b.text for b in response.content if b.type == "text"), "").strip()


_DEFAULT: Optional[LLMContextualizer] = None


def contextualize_llm(chunk: Chunk, all_chunks_in_doc: List[Chunk]) -> str:
    """Module-level convenience matching the ``contextualize`` signature."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMContextualizer()
    return _DEFAULT.generate(chunk, all_chunks_in_doc)
