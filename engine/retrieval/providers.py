"""Provider selection for contextualizers and embedders.

The default path is fully offline via HeuristicContextualizer.
Optional providers (Ollama, remote LLM, sentence-transformers) are stubs
that raise RuntimeError with opt-in instructions; any third-party import
is lazy (inside the method that needs it).

Provider selection: set env var RAG_PROVIDER to one of:
    heuristic  (default)
    ollama
    remote
    sentence_transformers
"""

import os
from typing import List

from retrieval.chunker import Chunk
from retrieval.contextualizer import contextualize as _heuristic_contextualize


class HeuristicContextualizer:
    """Default offline contextualizer.  No network, no model, no deps."""

    def generate(self, chunk: Chunk, all_chunks: List[Chunk]) -> str:
        return _heuristic_contextualize(chunk, all_chunks)


class OllamaContextualizer:
    """Stub: requires Ollama running locally + the 'requests' package."""

    def generate(self, chunk: Chunk, all_chunks: List[Chunk]) -> str:
        # Lazy import — never executed in the default path
        try:
            import requests  # noqa: F401  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "provider 'ollama' requires opt-in + dependency 'requests'; "
                "default path is offline"
            ) from exc
        raise RuntimeError(
            "provider 'ollama' requires opt-in + dependency 'requests'; "
            "default path is offline"
        )


class RemoteLLMContextualizer:
    """Stub: requires network access + API key + an HTTP client package."""

    def generate(self, chunk: Chunk, all_chunks: List[Chunk]) -> str:
        # Lazy import — never executed in the default path
        try:
            import httpx  # noqa: F401  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "provider 'remote' requires opt-in + dependency 'httpx'; "
                "default path is offline"
            ) from exc
        raise RuntimeError(
            "provider 'remote' requires opt-in + dependency 'httpx'; "
            "default path is offline"
        )


class SentenceTransformerEmbedder:
    """Stub: requires the 'sentence-transformers' package."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Lazy import — never executed in the default path
        try:
            import sentence_transformers  # noqa: F401  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "provider 'sentence_transformers' requires opt-in + dependency "
                "'sentence-transformers'; default path is offline"
            ) from exc
        raise RuntimeError(
            "provider 'sentence_transformers' requires opt-in + dependency "
            "'sentence-transformers'; default path is offline"
        )


def get_contextualizer():
    """Return a contextualizer/embedder based on RAG_PROVIDER env var (default 'heuristic').

    For 'heuristic', 'ollama', and 'remote' the returned object exposes a
    ``generate(chunk, all_chunks) -> str`` interface (contextualizer contract).

    For 'sentence_transformers' the returned object is a
    ``SentenceTransformerEmbedder`` that exposes an
    ``embed(texts) -> List[List[float]]`` interface (embedder contract).
    Construction succeeds even when the package is not installed; the
    ``RuntimeError`` is deferred until ``embed()`` is actually called.
    """
    provider = os.environ.get("RAG_PROVIDER", "heuristic").lower()
    if provider == "heuristic":
        return HeuristicContextualizer()
    elif provider == "ollama":
        return OllamaContextualizer()
    elif provider == "remote":
        return RemoteLLMContextualizer()
    elif provider == "sentence_transformers":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(
            f"Unknown RAG_PROVIDER '{provider}'. "
            "Choose from: heuristic, ollama, remote, sentence_transformers"
        )
