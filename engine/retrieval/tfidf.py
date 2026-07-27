"""TF-IDF index with log-normalized TF, smoothed IDF, and L2-normalized vectors.

Documents are stored as **sparse** vectors (term -> weight) with an inverted
index (term -> postings), so memory is O(total tokens) rather than
O(docs x vocab), and a query only scores documents that share a term with it.
Scores are identical to a dense cosine (non-shared terms contribute 0); the full
corpus is still returned, with zero-overlap documents scored 0.0.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

#: Supported tokenizer modes (see ``tokenize``).
TOKENIZER_MODES = ("plain", "code")

#: Splits an identifier into subtokens on separators (``_``, ``-``, ``.``) and
#: at camelCase/PascalCase boundaries (lower-to-upper, and the last upper of
#: an acronym run before a following Titlecase word, e.g. "HTTPSConnection"
#: -> "HTTPS" / "Connection").
_SPLIT_RE = re.compile(r"[_\-.]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str, mode: str = "plain") -> List[str]:
    """Shared tokenizer: lowercase, extract \\w+ tokens.

    ``mode="plain"`` (default) is byte-identical to the original tokenizer.
    ``mode="code"`` additionally emits each whole token's lowercased
    subtokens (split on ``_``/``-``/``.`` and camelCase/PascalCase
    boundaries) right after the whole token, left-to-right — e.g.
    ``"getUserById"`` -> ``["getuserbyid", "get", "user", "by", "id"]`` — but
    only when splitting actually yields >=2 non-empty parts, and never
    re-emitting a subtoken identical to the whole token (so a
    non-splitting token like ``"chunk"`` isn't duplicated).
    """
    if mode not in TOKENIZER_MODES:
        raise ValueError(f"unknown tokenizer mode {mode!r}; choose from {TOKENIZER_MODES}")
    if mode == "plain":
        return re.findall(r"\w+", text.lower())

    # Case-sensitive extraction first: camelCase/PascalCase boundaries only
    # exist in the original casing, so lowercasing up front (as "plain" does)
    # would destroy the very signal "code" mode splits on.
    tokens: List[str] = []
    for raw in re.findall(r"\w+", text):
        whole = raw.lower()
        tokens.append(whole)
        parts = [p for p in _SPLIT_RE.split(raw) if p]
        if len(parts) < 2:
            continue
        for part in parts:
            lowered = part.lower()
            if lowered != whole:
                tokens.append(lowered)
    return tokens


def _doc_weights(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Sparse, L2-normalized TF-IDF weights for one document.

    Weight(term) = (1 + log(tf)) * idf(term), then the whole vector is
    L2-normalized.  ``idf`` is precomputed once per corpus (it depends only on
    the term), so the per-document loop does no IDF math.  Matches the previous
    dense computation.
    """
    tf_counts = Counter(tokens)
    weights: Dict[str, float] = {}
    for w, count in tf_counts.items():
        tf = 1.0 + math.log(count) if count > 0 else 0.0
        weights[w] = tf * idf[w]
    norm = math.sqrt(sum(v * v for v in weights.values()))
    if norm > 0.0:
        for w in weights:
            weights[w] /= norm
    return weights


class TfidfIndex:
    """TF-IDF index over a corpus of documents.

    Parameters
    ----------
    tokenizer : str
        Tokenizer mode, one of ``TOKENIZER_MODES`` (default ``"plain"``).
        Raises ``ValueError`` on an unknown mode. The mode is persisted
        (``to_dict``/``from_dict``) and never re-derived at query time — a
        restored index always queries with the mode it was fit with.
    """

    def __init__(self, tokenizer: str = "plain") -> None:
        if tokenizer not in TOKENIZER_MODES:
            raise ValueError(
                f"unknown tokenizer mode {tokenizer!r}; choose from {TOKENIZER_MODES}"
            )
        self.tokenizer = tokenizer
        self._doc_vectors: List[Dict[str, float]] = []        # per-doc sparse weights
        self._postings: Dict[str, List[Tuple[int, float]]] = {}  # term -> [(doc, weight)]
        self._n_docs: int = 0

    def fit(self, docs: List[str]) -> None:
        """Build sparse TF-IDF vectors + an inverted index for a list of docs."""
        self._doc_vectors = []
        self._postings = {}

        if not docs:
            self._n_docs = 0
            return

        self._n_docs = len(docs)
        tokenized = [tokenize(d, self.tokenizer) for d in docs]

        df: Dict[str, int] = {}
        for tokens in tokenized:
            for w in set(tokens):
                df[w] = df.get(w, 0) + 1

        # IDF depends only on the term — compute it once, not per occurrence.
        idf = {w: math.log((self._n_docs + 1) / (dfw + 1)) + 1.0 for w, dfw in df.items()}

        for doc_idx, tokens in enumerate(tokenized):
            weights = _doc_weights(tokens, idf)
            self._doc_vectors.append(weights)
            for term, weight in weights.items():
                self._postings.setdefault(term, []).append((doc_idx, weight))

    def query(self, text: str) -> List[Tuple[int, float]]:
        """Return (doc_idx, cosine_score) pairs sorted descending."""
        if self._n_docs == 0:
            return []

        # Query vector: log-normalized TF, L2-normalized (no IDF, as before).
        tf_counts = Counter(tokenize(text, self.tokenizer))
        qvec: Dict[str, float] = {}
        for w, count in tf_counts.items():
            qvec[w] = 1.0 + math.log(count) if count > 0 else 0.0
        norm = math.sqrt(sum(v * v for v in qvec.values()))
        if norm > 0.0:
            for w in qvec:
                qvec[w] /= norm

        # Accumulate cosine via the inverted index: only docs sharing a term.
        scores: Dict[int, float] = {}
        for term, qweight in qvec.items():
            for doc_idx, dweight in self._postings.get(term, ()):  # noqa: B905
                scores[doc_idx] = scores.get(doc_idx, 0.0) + qweight * dweight

        # Zero-fill untouched docs to preserve the full ranking + tie order.
        ranked = [(idx, scores.get(idx, 0.0)) for idx in range(self._n_docs)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted state to a JSON-safe dict (tuples become lists)."""
        return {
            "tokenizer": self.tokenizer,
            "n_docs": self._n_docs,
            "doc_vectors": self._doc_vectors,
            "postings": {
                term: [[doc_idx, weight] for doc_idx, weight in postings]
                for term, postings in self._postings.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TfidfIndex":
        """Rebuild a ``TfidfIndex`` from ``to_dict`` output (re-tuples postings).

        The tokenizer mode is restored from the persisted dict (defaulting
        to ``"plain"`` for a pre-tokenizer-modes cache) — query time never
        re-derives it.
        """
        index = cls(tokenizer=data.get("tokenizer", "plain"))
        index._n_docs = data["n_docs"]
        index._doc_vectors = data["doc_vectors"]
        index._postings = {
            term: [(int(doc_idx), float(weight)) for doc_idx, weight in postings]
            for term, postings in data["postings"].items()
        }
        return index
