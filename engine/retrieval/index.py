"""ContextualRetriever: builds TF-IDF + BM25 over chunks and fuses results with RRF."""

from dataclasses import dataclass
from typing import Callable, List, Optional

from retrieval.bm25 import BM25Index
from retrieval.chunker import Chunk
from retrieval.contextualizer import contextualize
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.tfidf import TfidfIndex


@dataclass
class Result:
    chunk: Chunk
    tfidf_score: float
    bm25_score: float
    fused_rank: int


class ContextualRetriever:
    """Index chunks with TF-IDF and BM25, fuse rankings via RRF."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._texts: List[str] = []
        self._tfidf = TfidfIndex()
        self._bm25 = BM25Index()

    def build(
        self,
        chunks: List[Chunk],
        use_context: bool = False,
        contextualizer: Optional[Callable[[Chunk, List[Chunk]], str]] = None,
    ) -> None:
        """Index chunks, optionally prepending context before indexing.

        Parameters
        ----------
        chunks :
            Flat list of Chunk objects across all documents.
        use_context :
            If True (and no ``contextualizer`` given), use the offline heuristic
            contextualizer.
        contextualizer :
            Optional ``(chunk, doc_chunks) -> str`` callable that overrides the
            heuristic — e.g. the LLM-based ``LLMContextualizer().generate``.
            When provided, ``use_context`` is implied.
        """
        self._chunks = list(chunks)

        strategy = contextualizer or (contextualize if use_context else None)

        if strategy is not None:
            # Group by doc_id so each chunk sees its document's other chunks.
            by_doc: dict = {}
            for ch in chunks:
                by_doc.setdefault(ch.doc_id, []).append(ch)

            self._texts = [strategy(ch, by_doc[ch.doc_id]) for ch in chunks]
        else:
            self._texts = [ch.text for ch in chunks]

        self._tfidf.fit(self._texts)
        self._bm25.fit(self._texts)

    def search(self, query: str, top_k: int = 5) -> List[Result]:
        """Query both indexes, fuse with RRF, return top_k Results."""
        tfidf_scores_raw = self._tfidf.query(query)
        bm25_scores_raw = self._bm25.query(query)

        # Build score lookup dicts
        tfidf_lookup = {doc_idx: score for doc_idx, score in tfidf_scores_raw}
        bm25_lookup = {doc_idx: score for doc_idx, score in bm25_scores_raw}

        # Extract ordered doc_idx lists for RRF
        tfidf_ranking = [doc_idx for doc_idx, _ in tfidf_scores_raw]
        bm25_ranking = [doc_idx for doc_idx, _ in bm25_scores_raw]

        fused = reciprocal_rank_fusion([tfidf_ranking, bm25_ranking])

        results: List[Result] = []
        for rank, (doc_idx, _fused_score) in enumerate(fused[:top_k]):
            chunk = self._chunks[doc_idx]
            results.append(
                Result(
                    chunk=chunk,
                    tfidf_score=tfidf_lookup.get(doc_idx, 0.0),
                    bm25_score=bm25_lookup.get(doc_idx, 0.0),
                    fused_rank=rank,
                )
            )
        return results
