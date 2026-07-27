"""BM25 Okapi index using the shared tokenizer from tfidf.py.

Uses an inverted index (term -> postings) so a query only touches documents that
contain a query term, instead of scanning the whole corpus per query.  Scores
are identical to a full scan; only documents with no query-term overlap (score
0) are filled in at the end to preserve the full ranking.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from retrieval.tfidf import TOKENIZER_MODES, tokenize


class BM25Index:
    """BM25 Okapi ranking over a corpus of documents.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalization parameter (default 0.75).
    tokenizer : str
        Tokenizer mode, one of ``retrieval.tfidf.TOKENIZER_MODES`` (default
        ``"plain"``). Raises ``ValueError`` on an unknown mode. Persisted
        (``to_dict``/``from_dict``) and never re-derived at query time.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: str = "plain") -> None:
        if tokenizer not in TOKENIZER_MODES:
            raise ValueError(
                f"unknown tokenizer mode {tokenizer!r}; choose from {TOKENIZER_MODES}"
            )
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self._tf: List[Dict[str, int]] = []      # per-doc term frequencies
        self._dl: List[int] = []                 # per-doc length
        self._postings: Dict[str, List[int]] = {}  # term -> doc indices
        self._df: Dict[str, int] = {}
        self._n_docs: int = 0
        self._avgdl: float = 1.0

    def fit(self, docs: List[str]) -> None:
        """Tokenize and index a list of document strings."""
        self._tf = []
        self._dl = []
        self._postings = {}
        self._df = {}

        if not docs:
            self._n_docs = 0
            return

        self._n_docs = len(docs)
        total_len = 0
        for doc_idx, doc in enumerate(docs):
            tokens = tokenize(doc, self.tokenizer)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._dl.append(len(tokens))
            total_len += len(tokens)
            for term in counts:  # one posting per (term, doc)
                self._postings.setdefault(term, []).append(doc_idx)

        self._df = {term: len(docs_with) for term, docs_with in self._postings.items()}
        self._avgdl = total_len / self._n_docs if self._n_docs else 1.0
        if self._avgdl == 0:
            self._avgdl = 1.0

    def _idf(self, term: str) -> float:
        """Smoothed IDF: log(1 + (N - df + 0.5) / (df + 0.5))."""
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))

    def query(self, text: str) -> List[Tuple[int, float]]:
        """Return (doc_idx, bm25_score) pairs sorted descending.

        Only documents containing at least one query term are scored; the rest
        receive 0.0, so the full corpus is still returned (matching a full scan).
        """
        if self._n_docs == 0:
            return []

        query_terms = set(tokenize(text, self.tokenizer))
        scores: Dict[int, float] = {}
        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_idx in postings:
                f = self._tf[doc_idx][term]
                dl = self._dl[doc_idx]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * f * (self.k1 + 1) / denom

        # Zero-fill untouched docs to preserve the full ranking + tie order.
        ranked = [(idx, scores.get(idx, 0.0)) for idx in range(self._n_docs)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted state to a JSON-safe dict (Counters become plain dicts)."""
        return {
            "k1": self.k1,
            "b": self.b,
            "tokenizer": self.tokenizer,
            "n_docs": self._n_docs,
            "avgdl": self._avgdl,
            "tf": [dict(counts) for counts in self._tf],
            "dl": self._dl,
            "postings": self._postings,
            "df": self._df,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        """Rebuild a ``BM25Index`` from ``to_dict`` output.

        Restored ``tf``/``postings``/``df`` stay plain dicts/lists — only
        ``[]`` indexing is used on them at query time, so no re-hydration
        into ``Counter`` is required. The tokenizer mode is restored from
        the persisted dict (defaulting to ``"plain"`` for a pre-tokenizer-
        modes cache) — query time never re-derives it.
        """
        index = cls(k1=data["k1"], b=data["b"], tokenizer=data.get("tokenizer", "plain"))
        index._n_docs = data["n_docs"]
        index._avgdl = data["avgdl"]
        index._tf = data["tf"]
        index._dl = data["dl"]
        index._postings = data["postings"]
        index._df = data["df"]
        return index
