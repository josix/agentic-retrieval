"""Signal-based hyperparameter auto-tuning heuristics.

Stdlib-only (``re``, ``statistics``, ``pathlib``, ``collections``,
``dataclasses``) — this module must import cleanly with zero optional
extras installed, same guarantee as the rest of the default pipeline. It
does **not** import ``retrieval.retrievers`` — auto-tuning only produces a
plain params dict; wiring that dict into an actual retriever construction
is the CLI's job (``retrieval.cli``), keeping this module decoupled from
the retriever-construction layer.

Two-pass signal collection (see ``collect_signals``):

* Pass 1 (pre-chunking): cheap filesystem stats — file count, extension
  histogram, byte counts, code-vs-non-code byte fraction, median code-file
  line count. Enough to pick a ``ChunkingPolicy`` and tokenizer mode.
* Pass 2 (post-chunking, using the pass-1-informed policy + tokenizer):
  actual chunk-token-count statistics (mean, coefficient of variation,
  percentiles) — needed for the length-normalization (``bm25_b``) and
  saturation-tail (``bm25_k1``) heuristics, and for the persisted
  ``corpus_stats`` block that query-time ``top_k`` resolution reads back.

All heuristics degrade gracefully to the engine's existing static defaults
on an empty corpus (no files, or no chunks) — see the individual
``_decide_*`` functions — so ``resolve_params`` never propagates a ``None``
into a caller-facing param.
"""

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from retrieval import project_loader
from retrieval.ast_chunker import LANGUAGE_BY_SUFFIX
from retrieval.chunker import ChunkingPolicy
from retrieval.tfidf import tokenize

#: chunk-token p10 below this is flagged (not auto-corrected) by ``tune``:
#: it usually means chunks are being sliced too finely for BM25/TF-IDF to
#: carry much signal per chunk.
_THIN_CHUNK_P10_THRESHOLD = 15


@dataclass
class CorpusSignals:
    """Raw corpus measurements ``resolve_params`` derives hyperparameters
    from. Pass-2 fields (``n_chunks`` onward) are ``None`` for an empty
    corpus (no discovered files, or no chunks produced)."""

    n_files: int
    ext_histogram: Dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    code_bytes: int = 0
    code_fraction: float = 0.0
    median_code_lines: Optional[float] = None
    n_chunks: Optional[int] = None
    mean_chunk_tokens: Optional[float] = None
    chunk_token_cv: Optional[float] = None
    p10_chunk_tokens: Optional[float] = None
    p50_chunk_tokens: Optional[float] = None
    p90_chunk_tokens: Optional[float] = None


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (0-100) over an already-sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def _decide_tokenizer(code_fraction: float) -> str:
    return "code" if code_fraction > 0.3 else "plain"


def _decide_chunking_policy(
    code_fraction: float, median_code_lines: Optional[float]
) -> ChunkingPolicy:
    code_chars = 1200 if code_fraction > 0.3 else 400
    ast_max_chars = 1200 if median_code_lines is None or median_code_lines <= 400 else 2000
    return ChunkingPolicy(
        prose_chars=400,
        config_chars=700,
        code_chars=code_chars,
        default_chars=400,
        ast_max_chars=ast_max_chars,
    )


def _decide_bm25_k1(code_fraction: float, median_chunk_tokens: Optional[float]) -> float:
    if code_fraction > 0.6:
        return 1.2
    if code_fraction >= 0.2:
        return 1.5
    if median_chunk_tokens is not None and median_chunk_tokens > 150:
        return 1.8
    return 1.5


def _decide_bm25_b(
    chunk_token_cv: Optional[float],
    p50_chunk_tokens: Optional[float],
    p90_chunk_tokens: Optional[float],
) -> float:
    if chunk_token_cv is None:
        return 0.75  # static default: no pass-2 chunk stats (empty corpus)
    spread_ratio = None
    if p50_chunk_tokens:
        spread_ratio = (p90_chunk_tokens or 0.0) / p50_chunk_tokens
    if chunk_token_cv > 1.2 or (spread_ratio is not None and spread_ratio > 4):
        return 1.0
    if chunk_token_cv >= 0.4:
        return 0.75
    return 0.3


#: sentence-transformers models ``_decide_embed_model`` chooses between,
#: mapped to their embedding dimensionality (``_decide_bit_width`` sizes the
#: quantized index from it). An unknown/override model falls back to 384.
_EMBED_MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "flax-sentence-embeddings/st-codesearch-distilroberta-base": 768,
}

#: chunk count at or below which the corpus is small enough that the slower,
#: higher-quality mpnet embedder is worth the per-chunk embedding cost.
_SMALL_CORPUS_CHUNKS = 2000


def _decide_embed_model(code_fraction: float, n_chunks: Optional[int]) -> str:
    """Turbovec/hybrid embedding model, from corpus content and size.

    Content first: a code-dominated corpus (>0.6 code bytes, same threshold
    as the ``bm25_k1`` code tier) gets a code-search-trained embedder —
    prose-trained MiniLM/mpnet embed identifiers poorly. Otherwise size
    decides the quality/cost tradeoff: embedding cost scales linearly with
    chunk count, so a small corpus (<= ``_SMALL_CORPUS_CHUNKS`` chunks)
    affords the higher-quality mpnet, while a large one keeps the fast
    MiniLM static default. No pass-2 chunk count at all (empty corpus)
    degrades to the MiniLM static default, like every other heuristic.
    """
    if n_chunks is None:
        # static default (empty corpus / no pass-2 stats) — must match the
        # TurbovecRetriever ctor default so the no-signal path is a no-op
        return "sentence-transformers/all-MiniLM-L6-v2"
    if code_fraction > 0.6:
        return "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    if n_chunks <= _SMALL_CORPUS_CHUNKS:
        return "sentence-transformers/all-mpnet-base-v2"
    return "sentence-transformers/all-MiniLM-L6-v2"


def _decide_bit_width(n_chunks: Optional[int], embed_dims: int = 384) -> int:
    # NOTE: the originally specified tiers were 8/4/2, but the installed
    # turbovec.TurboQuantIndex only accepts bit_width in {2, 3, 4} (raises
    # ValueError otherwise — confirmed empirically, see the deviation noted
    # in the implementation report). Remapped to 4/3/2 here, preserving the
    # intended ordering (smaller corpus -> higher precision).
    if not n_chunks:
        return 4  # static default: no pass-2 chunk count (empty corpus)
    total = n_chunks * embed_dims
    if total <= 5e7:
        return 4
    if total <= 5e8:
        return 3
    return 2


def _decide_lucene_params(median_chunk_tokens: Optional[float]) -> Tuple[float, float]:
    """Pi-serini (Lucene BM25) ``k1``/``b``, chosen by median chunk length
    in tokens: MS MARCO passage tuning (``0.9``/``0.4``) for short units
    (median <=2000 tokens, including an unknown/empty-corpus median), else
    the long-document tuning (``25``/``1``) meant for >2000-token units.

    Today this always resolves to ``0.9``/``0.4``: this engine's chunkers
    cap chunk size around 1200 characters (~1200 tokens at most, well
    under the 2000-token threshold), so the long-document branch is
    presently unreachable — recorded explicitly here (rather than omitted
    from ``resolve_params``) so that decision is visible in persisted meta
    instead of silently defaulting inside ``PiSeriniRetriever``.
    """
    if median_chunk_tokens is not None and median_chunk_tokens > 2000:
        return 25.0, 1.0
    return 0.9, 0.4


def resolve_top_k(corpus_stats: Optional[Dict[str, Any]]) -> int:
    """Query-time ``top_k`` from a persisted ``corpus_stats`` block:
    ``clamp(round(1500 / mean_chunk_tokens), 5, 30)``, falling back to 5 on
    a missing/zero/absent ``mean_chunk_tokens`` (empty-corpus guard)."""
    mean_tokens = (corpus_stats or {}).get("mean_chunk_tokens")
    if not mean_tokens:
        return 5
    return max(5, min(30, round(1500 / mean_tokens)))


def collect_signals(
    root: "Path | str", *, tokenizer_for_pass2: Optional[str] = None
) -> CorpusSignals:
    """Two-pass ``CorpusSignals`` collection over *root* (see module
    docstring). *tokenizer_for_pass2* overrides the pass-1-derived
    tokenizer mode used to token-count pass-2 chunks (mainly for tests);
    left ``None``, pass 2 uses whatever pass 1's ``code_fraction`` implies.
    """
    root = Path(root)
    files = project_loader.discover_files(root)
    ext_histogram: Counter = Counter(f.suffix.lower() for f in files)

    total_bytes = 0
    code_bytes = 0
    code_line_counts: List[int] = []
    for file_path in files:
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        total_bytes += size
        if file_path.suffix.lower() in LANGUAGE_BY_SUFFIX:
            code_bytes += size
            text = project_loader.read_text_safe(file_path)
            if text is not None:
                code_line_counts.append(text.count("\n") + 1)

    code_fraction = (code_bytes / total_bytes) if total_bytes else 0.0
    median_code_lines = statistics.median(code_line_counts) if code_line_counts else None

    if not files:
        return CorpusSignals(n_files=0, ext_histogram={})

    pass1_policy = _decide_chunking_policy(code_fraction, median_code_lines)
    tokenizer_mode = tokenizer_for_pass2 or _decide_tokenizer(code_fraction)

    chunk_documents = project_loader.load_chunk_documents(root, policy=pass1_policy)
    chunk_token_counts = sorted(len(tokenize(d.text, tokenizer_mode)) for d in chunk_documents)

    signals = CorpusSignals(
        n_files=len(files),
        ext_histogram=dict(ext_histogram),
        total_bytes=total_bytes,
        code_bytes=code_bytes,
        code_fraction=code_fraction,
        median_code_lines=median_code_lines,
    )
    if not chunk_token_counts:
        return signals

    mean_tokens = statistics.mean(chunk_token_counts)
    stdev_tokens = statistics.pstdev(chunk_token_counts) if len(chunk_token_counts) > 1 else 0.0
    signals.n_chunks = len(chunk_token_counts)
    signals.mean_chunk_tokens = mean_tokens
    signals.chunk_token_cv = (stdev_tokens / mean_tokens) if mean_tokens else 0.0
    signals.p10_chunk_tokens = _percentile(chunk_token_counts, 10)
    signals.p50_chunk_tokens = _percentile(chunk_token_counts, 50)
    signals.p90_chunk_tokens = _percentile(chunk_token_counts, 90)
    return signals


def thin_chunk_warning(signals: CorpusSignals) -> Optional[str]:
    """Return a warning string if pass-2 chunk-token p10 is suspiciously
    thin (<15 tokens), else ``None``. Advisory only — ``resolve_params``
    never auto-corrects for this."""
    p10 = signals.p10_chunk_tokens
    if p10 is not None and p10 < _THIN_CHUNK_P10_THRESHOLD:
        return (
            f"chunk-token p10 ({p10:.1f}) is below {_THIN_CHUNK_P10_THRESHOLD} — "
            "chunks may be sliced too finely for BM25/TF-IDF to carry much "
            "signal per chunk (no auto-correction applied)"
        )
    return None


def resolve_params(
    signals: CorpusSignals, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Derive the full agent-decidable hyperparameter dict from *signals*,
    then apply *overrides* (any key with a non-``None`` value always wins —
    the "explicit > auto-derived > static default" precedence).

    Deterministic: two calls with the same *signals* return equal dicts.
    """
    policy = _decide_chunking_policy(signals.code_fraction, signals.median_code_lines)
    lucene_k1, lucene_b = _decide_lucene_params(signals.p50_chunk_tokens)
    # An explicit model override participates early so bit_width is sized
    # from the dimensionality of the model actually used, not the auto pick.
    model_name = (overrides or {}).get("model_name") or _decide_embed_model(
        signals.code_fraction, signals.n_chunks
    )
    params: Dict[str, Any] = {
        **policy.to_dict(),
        "tokenizer": _decide_tokenizer(signals.code_fraction),
        "bm25_k1": _decide_bm25_k1(signals.code_fraction, signals.p50_chunk_tokens),
        "bm25_b": _decide_bm25_b(
            signals.chunk_token_cv, signals.p50_chunk_tokens, signals.p90_chunk_tokens
        ),
        "model_name": model_name,
        "bit_width": _decide_bit_width(
            signals.n_chunks, _EMBED_MODEL_DIMS.get(model_name, 384)
        ),
        "lucene_k1": lucene_k1,
        "lucene_b": lucene_b,
    }
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
    return params
