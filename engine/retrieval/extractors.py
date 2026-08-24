"""Sidecar-transcript extraction for non-text-native document formats (PDF +
agent-only media).

Stdlib-only at import scope (``dataclasses``, ``hashlib``, ``io``, ``json``,
``os``, ``re``, ``struct``, ``sys``, ``time``, ``statistics``,
``unicodedata``, ``pathlib``, ``typing``) — this module must import cleanly
with zero optional extras installed, same guarantee as the rest of the
default pipeline. The ``pypdf`` backend is imported lazily, only inside the
functions that actually need it, mirroring the guidance-``RuntimeError``
convention used by ``retrieval.retrievers._require_backends`` and
``retrieval.ast_chunker.chunk_code``.

This module deliberately does **not** import ``retrieval.persistence`` or
``retrieval.project_loader``: ``persistence`` imports ``project_loader``,
which imports this module — importing either back would create a cycle.
``_CACHE_DIRNAME`` therefore duplicates ``persistence.CACHE_DIRNAME``'s value
as its own constant (see the drift-guard test in ``tests/test_extractors.py``
that asserts the two stay equal).

Design: a PDF's extracted text is written once to a Markdown "sidecar" file
under ``<project-root>/.agentic-retrieval/extracted/<rel-path>.md`` (always
under the project root, deliberately ignoring ``RETRIEVAL_INDEX_DIR`` — a
sidecar is a citation target a coding agent ``Read()``s by project-relative
path, so it must live in the tree being indexed even when the *cache* itself
is redirected elsewhere). A content-hash + extractor-version manifest
(``manifest.json``, alongside the sidecars) makes re-extraction a no-op on
unchanged files across the multiple loader passes ``index --auto`` performs
per run. Every failure mode (encrypted, malformed, empty, no-text-layer,
backend-missing) still produces a non-empty, human-readable sidecar — the
chunker returns zero chunks for blank input, so an empty stub would silently
vanish from every index.

No-install alternative: when ``pypdf`` isn't installed (or installing it
isn't an option), ``register_sidecar`` lets a coding agent that read the
PDF itself hand-author the transcript and register it directly — never
importing ``pypdf`` on that path either. An agent-authored entry is a
durable, first-class sidecar (``AGENT_EXTRACTOR_VERSION``), never silently
superseded by a later ``pypdf`` install; only an explicit ``retrieval
extract --force`` overwrites it. See ``retrieval sidecar --list
--register`` (``retrieval.cli``) and the ``retrieval`` skill's "Without the
pdf extra" section.

Beyond PDF (and caption files, ``.srt``/``.vtt`` — a second stdlib-only
machine extractor, see ``_extract_captions``), a second tier of media
(``AGENT_ONLY_EXTENSIONS`` — office docs and images) has **no machine
extractor at all**: ``ensure_sidecar`` always writes an ``"agent-only"``
stub for these suffixes, regardless of what's installed, and the
``retrieval sidecar --register`` workflow above is the *only* way to index
their real content.

A third tier (``AGENT_ORCHESTRATED_EXTENSIONS`` — audio/video) goes further
still: an agent cannot even read these natively, so it must orchestrate an
external tool (ASR via Bash) and register the result the same way. Because
these files can be arbitrarily large, ``ensure_sidecar``/``register_sidecar``
never read their bytes at all for identity purposes — see
``_source_identity``'s stat-only ``"stat/1"`` hash for this tier, versus the
byte-hash ``"sha256/1"`` every other tier uses.
"""

import hashlib
import io
import json
import os
import re
import statistics
import struct
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Tuple

#: Identifies the extraction algorithm + output format; bumping this string
#: invalidates every manifest entry (and therefore every cached sidecar),
#: forcing full re-extraction — the manifest equivalent of a schema bump.
EXTRACTOR_VERSION = "pypdf-text/1"

#: Extractor-version string stamped on manifest entries written by
#: ``register_sidecar`` (an agent-authored transcript, not a pypdf-parsed
#: one). Distinct from ``EXTRACTOR_VERSION`` so the two provenances never
#: collide in the manifest, but both are accepted by ``_is_cache_hit`` (see
#: ``_ACCEPTED_EXTRACTOR_VERSIONS``) — an agent transcript is a durable,
#: first-class sidecar, not a stub awaiting a real extraction.
AGENT_EXTRACTOR_VERSION = "agent-authored/1"

#: Per-suffix extractor-version overrides, keyed by lowercased suffix (e.g.
#: ``{".srt": "captions/1"}``). ``EXTRACTOR_VERSION`` remains the top-level
#: manifest ``extractor_version`` field (a single schema-bump string for the
#: whole manifest); this dict lets a *specific* extractor (e.g. the caption
#: converter) stamp its own version onto its own entries without bumping
#: every other suffix's cache. Populated by ``register_extractor``'s
#: ``version=`` kwarg. See ``extractor_version_for``.
_EXTRACTOR_VERSIONS: Dict[str, str] = {}

#: Extractor-version values ``_is_cache_hit`` treats as fresh; membership
#: (not equality with ``EXTRACTOR_VERSION``) so an agent-authored entry
#: survives across ``ensure_sidecar`` calls instead of being treated as
#: stale pypdf output. Rebuilt (not reassigned) whenever ``_EXTRACTOR_VERSIONS``
#: changes, via ``register_extractor``.
_ACCEPTED_EXTRACTOR_VERSIONS = frozenset({EXTRACTOR_VERSION, AGENT_EXTRACTOR_VERSION})

#: ``manifest["entries"][rel]["authored_by"]`` value for an agent-registered
#: sidecar (see ``register_sidecar``).
AUTHORED_BY_AGENT = "agent"

#: Suffixes whose machine extractor needs the optional ``pypdf`` backend
#: (currently just ``.pdf``). Distinct from ``MACHINE_EXTRACTABLE_EXTENSIONS``
#: (below): the latter is "has *some* registered extractor" (pypdf-backed
#: *or* stdlib-only, e.g. captions), while this set is specifically "needs a
#: backend install to produce real output" — the question ``require_extractors``
#: and ``ensure_sidecar``'s backend-missing-stub branch actually need to ask.
PYPDF_EXTENSIONS = frozenset({".pdf"})

#: Suffixes with a registered machine extractor (pypdf-backed ``.pdf``, and
#: stdlib-only ones like caption files — see ``PYPDF_EXTENSIONS`` for the
#: subset needing a backend). Used where "can a machine attempt real
#: extraction" is the question (``require_extractors`` preflight,
#: ``cli._discover_machine_extractable`` / ``_cmd_extract``'s extraction
#: loop) — never for discovery eligibility, which is
#: ``EXTRACTABLE_EXTENSIONS`` (below).
MACHINE_EXTRACTABLE_EXTENSIONS = frozenset({".pdf", ".srt", ".vtt"})

#: Suffixes with **no** machine extractor at all: office documents and
#: images a coding agent can read/view natively (Read tool / vision) but
#: this module can never parse itself. ``ensure_sidecar`` always writes an
#: ``"agent-only"`` stub for these — the sidecar-register workflow is the
#: only indexing path. See ``AGENT_ORCHESTRATED_EXTENSIONS`` below for the
#: third tier (audio/video), which is NOT this: those need an external tool
#: the agent runs, not something the agent reads/views itself.
AGENT_ONLY_EXTENSIONS = frozenset(
    {".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
)

#: Third media tier: audio/video suffixes with no machine extractor AND no
#: native agent-readable path either (unlike ``AGENT_ONLY_EXTENSIONS`` — an
#: agent can ``Read()``/view a docx or png directly, but it cannot "read"
#: an mp4's bytes and produce a transcript). The defining property: an agent
#: indexes these by *orchestrating an external tool* (ASR via Bash — see
#: ``_MSG_AGENT_ORCHESTRATED``), never by reading the source natively.
#: ``ensure_sidecar`` never reads these files' bytes at all (see
#: ``_source_identity``) — only ``os.stat``, deliberately, since these files
#: can be arbitrarily large and the engine has no business allocating a
#: multi-GB buffer just to compute a hash it then discards.
AGENT_ORCHESTRATED_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".m4a", ".wav", ".flac"}
)

#: File suffixes this module can produce a sidecar for (PDF/captions +
#: agent-only + agent-orchestrated media) — either via a machine extractor
#: or via ``register_sidecar``. ``project_loader`` discovery,
#: ``register_sidecar``'s suffix validation, and
#: ``persistence.compute_fingerprint`` all key off this union, so a new
#: suffix added to any tier is discovered/fingerprinted for free.
EXTRACTABLE_EXTENSIONS = (
    MACHINE_EXTRACTABLE_EXTENSIONS | AGENT_ONLY_EXTENSIONS | AGENT_ORCHESTRATED_EXTENSIONS
)

#: Raw source-file byte cap applied at discovery time for extractable
#: suffixes (see ``project_loader.discover_files``'s ``extract_max_bytes``).
EXTRACT_MAX_BYTES = 25_000_000

#: Post-extraction transcript character budget; a longer transcript is
#: truncated at the last whole-page boundary under this limit (see
#: ``_truncate_at_unit_boundary``).
MAX_TRANSCRIPT_CHARS = 1_000_000

#: Duplicates ``retrieval.persistence.CACHE_DIRNAME``'s value — this module
#: cannot import ``persistence`` (circular-import hazard: persistence ->
#: project_loader -> extractors). Kept in sync by a drift-guard test.
_CACHE_DIRNAME = ".agentic-retrieval"

#: Per-page memory guard: a content stream larger than this is skipped
#: (placeholder emitted instead) rather than decompressed, per pypdf's own
#: guidance that a pathological stream can inflate to gigabytes in memory.
_PAGE_MEMORY_GUARD_BYTES = 50_000_000

_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")
_WS_COLLAPSE_RE = re.compile(r"[ \t]{2,}")

#: Matches a section-heading line in a sidecar transcript — either the PDF
#: convention (``## Page N``) or the time-coded convention used by caption/
#: media transcripts (``## [HH:MM:SS] label``). Used both to count units
#: (pages, or timestamped sections) in an agent-authored transcript and to
#: find whole-unit boundaries for ``register_sidecar``'s oversized-transcript
#: truncation. Deliberately narrow to these two shapes — a generic ``## ...``
#: match would treat prose subheadings as unit boundaries.
_UNIT_HEADING_RE = re.compile(
    r"^## (?:Page \d+|\[\d{2}:\d{2}:\d{2}\][^\n]*)\s*$", re.MULTILINE
)

#: Strips a leading run of ``<!-- ... -->`` comment lines from an
#: agent-authored transcript before rendering — ``_render_sidecar`` always
#: writes its own header, so a re-registered transcript that (redundantly)
#: includes one from a prior read must not end up with two.
_LEADING_COMMENT_RE = re.compile(r"\A(?:<!--[^\n]*-->\n)+")


@dataclass(frozen=True)
class Extraction:
    """Raw extraction result for one PDF's bytes (backend-agnostic)."""

    text: str
    status: str
    reason: str
    pages: int
    truncated: bool


@dataclass(frozen=True)
class Sidecar:
    """A resolved, on-disk sidecar — what ``ensure_sidecar`` returns."""

    docid: str
    path: Path
    text: str
    status: str
    reason: str
    cache_hit: bool


_EXTRACTORS: Dict[str, Callable[[bytes], Extraction]] = {}

#: Process-level memo of ``ensure_sidecar`` results, keyed by
#: ``(abs_path, st_size, st_mtime_ns)`` — avoids re-hashing/re-extracting the
#: same file multiple times within one ``index --auto`` run (which loads the
#: corpus up to 7 times). Cleared via ``clear_process_cache`` (test hook).
_PROCESS_MEMO: Dict[Tuple[str, int, int], Sidecar] = {}

#: Set once a PDF has been processed without a usable backend, so the
#: "install the pdf extra" warning prints at most once per process.
_WARNED = False

#: Set once an agent-only-media file has been stubbed, so the
#: "register a transcript" note prints at most once per process (mirrors
#: ``_WARNED``).
_WARNED_AGENT_ONLY = False

#: Set once a tier-3 agent-orchestrated (audio/video) file has been
#: stubbed, so its distinct "run ASR, then register" note prints at most
#: once per process (mirrors ``_WARNED_AGENT_ONLY``).
_WARNED_AGENT_ORCHESTRATED = False

_MSG_ENCRYPTED = (
    "This PDF is password-protected and could not be decrypted with an "
    "empty password, so no text could be extracted."
)
_MSG_CRYPTO_UNAVAILABLE = (
    "This PDF uses an encryption method that requires the optional "
    "'cryptography' package, which is not installed, so no text could be "
    "extracted."
)
_MSG_EMPTY = "This PDF has zero pages, so there is no text to extract."
_MSG_NO_TEXT_LAYER = (
    "This PDF appears to be a scanned or image-only document with no "
    "extractable text layer (OCR is not performed by this extractor)."
)
_MSG_MALFORMED = (
    "This PDF could not be parsed; it may be corrupted or use an unsupported format."
)
_MSG_BACKEND_MISSING = (
    "pypdf is not installed, so this PDF's text could not be extracted. "
    "Install the 'pdf' extra (`uv pip install -e '.[pdf]'`) and reindex to "
    "extract real content. Alternatively, an agent can read this PDF itself "
    "and author the transcript directly via `retrieval sidecar --register "
    "<pdf> --transcript <file>` — no pypdf install required."
)
_MSG_AGENT_ONLY = (
    "This file's format has no machine extractor; no text has been "
    "extracted automatically. A coding agent can read the original file "
    "natively (Read tool for documents, vision for images) and register a "
    "transcript directly via `retrieval sidecar --register <file> "
    "--transcript <transcript> --root <root>`."
)
_MSG_AGENT_ORCHESTRATED = (
    "This audio/video file has no machine extractor, and a coding agent "
    "cannot read it natively either (unlike documents/images). Run an ASR "
    "tool via Bash (e.g. WhisperX, whisper.cpp, or whisper) to transcribe "
    "it, then register the result directly via `retrieval sidecar "
    "--register <file> --transcript <transcript> --root <root>`. If no ASR "
    "tool is available, leave this stub as-is — it is an honest state, not "
    "an error."
)


def register_extractor(
    suffix: str, fn: Callable[[bytes], Extraction], *, version: str | None = None
) -> None:
    """Register *fn* as the extractor for files with *suffix* (e.g. ``.pdf``).

    *version*, when given, is recorded in ``_EXTRACTOR_VERSIONS`` and folded
    into ``_ACCEPTED_EXTRACTOR_VERSIONS`` — a non-pypdf extractor (e.g. the
    caption converter, ``captions/1``) stamps its own manifest-entry
    ``extractor_version`` via ``extractor_version_for`` instead of the
    top-level ``EXTRACTOR_VERSION`` (which stays pypdf's). Omitting *version*
    (the ``.pdf`` registration below) leaves ``EXTRACTOR_VERSION`` as that
    suffix's version, unchanged from prior releases.
    """
    global _ACCEPTED_EXTRACTOR_VERSIONS
    _EXTRACTORS[suffix.lower()] = fn
    if version is not None:
        _EXTRACTOR_VERSIONS[suffix.lower()] = version
        _ACCEPTED_EXTRACTOR_VERSIONS = frozenset(
            {EXTRACTOR_VERSION, AGENT_EXTRACTOR_VERSION} | set(_EXTRACTOR_VERSIONS.values())
        )


def extractor_for(path: "os.PathLike[str] | str") -> Callable[[bytes], Extraction] | None:
    """Return the registered extractor for *path*'s suffix, or ``None``."""
    return _EXTRACTORS.get(Path(path).suffix.lower())


def extractor_version_for(path: "os.PathLike[str] | str") -> str:
    """Return the manifest ``extractor_version`` string to stamp for
    *path*'s suffix: its ``_EXTRACTOR_VERSIONS`` override if one was
    registered (e.g. ``captions/1`` for ``.srt``/``.vtt``), else the
    top-level ``EXTRACTOR_VERSION`` (pypdf's, unchanged default)."""
    return _EXTRACTOR_VERSIONS.get(Path(path).suffix.lower(), EXTRACTOR_VERSION)


def require_extractors(extensions: Iterable[str]) -> None:
    """Raise a guidance ``RuntimeError`` if *extensions* needs the ``pypdf``
    backend and it isn't installed (mirrors ``retrieval.retrievers``'
    guidance-``RuntimeError`` convention: ``RuntimeError``, not
    ``ImportError``, two-line message).

    Only checks extension membership in ``PYPDF_EXTENSIONS`` (currently
    ``.pdf``) — never probes import success for eligibility/discovery, only
    for this explicit preflight call. Every other extractable suffix
    (``AGENT_ONLY_EXTENSIONS``, stdlib-only machine extractors like
    captions, ``AGENT_ORCHESTRATED_EXTENSIONS``) never needs a backend, so
    they're excluded from this guard — this is a deliberate split from the
    former "any ``MACHINE_EXTRACTABLE_EXTENSIONS`` member needs pypdf"
    conflation, which would have misreported a caption-only tree as needing
    ``pypdf``.
    """
    if not any(ext.lower() in PYPDF_EXTENSIONS for ext in extensions):
        return
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - guidance path
        raise RuntimeError(
            "PDF extraction needs the 'pdf' extra:\n"
            "  uv pip install -e '.[pdf]'\n"
            "No-install alternative: an agent can author the transcript "
            "itself and register it with `retrieval sidecar --register "
            "<pdf> --transcript <file>`."
        ) from exc


def backend_available() -> bool:
    """Lazy-import probe for the ``pypdf`` backend.

    Used ONLY for content regeneration / staleness decisions (see
    ``ensure_sidecar``'s cache-hit check and ``needs_reextraction``) — NEVER
    for file-discovery eligibility, which must stay deterministic regardless
    of which optional extras happen to be installed in a given environment.
    """
    try:
        import pypdf  # noqa: F401
    except ImportError:
        return False
    return True


def clear_process_cache() -> None:
    """Clear the process-level ``ensure_sidecar`` memo (test hook)."""
    _PROCESS_MEMO.clear()


def invalidate_process_cache(source: "os.PathLike[str] | str") -> None:
    """Drop every ``_PROCESS_MEMO`` entry for *source*, regardless of the
    ``(size, mtime_ns)`` it was memoized under.

    ``register_sidecar`` calls this after writing a new manifest entry: the
    process memo is keyed by ``(abs_path, st_size, st_mtime_ns)`` of the
    *source* PDF, which registration doesn't touch, so a stale ``stub``
    result from an earlier ``ensure_sidecar`` call in this same process
    would otherwise keep being served instead of the freshly registered
    transcript.
    """
    resolved = str(Path(source).resolve())
    for key in [k for k in _PROCESS_MEMO if k[0] == resolved]:
        del _PROCESS_MEMO[key]


def extract_dir(root: "os.PathLike[str] | str") -> Path:
    """Sidecar directory for *root*: always ``<root>/.agentic-retrieval/extracted``.

    Deliberately ignores ``RETRIEVAL_INDEX_DIR`` (unlike
    ``persistence.index_dir``): a sidecar is a citation target a coding agent
    ``Read()``s by project-relative path, so it must live inside the project
    tree being indexed even when the index *cache* itself is redirected to a
    shared external directory.
    """
    return Path(root).resolve() / _CACHE_DIRNAME / "extracted"


def sidecar_relpath(rel_source: str) -> str:
    """Return the project-relative sidecar path for *rel_source*."""
    return f"{_CACHE_DIRNAME}/extracted/{rel_source}.md"


def _manifest_path(root: "os.PathLike[str] | str") -> Path:
    return extract_dir(root) / "manifest.json"


def load_manifest(root: "os.PathLike[str] | str") -> Dict[str, Any]:
    """Load the extraction manifest for *root*; missing/corrupt -> ``{}``."""
    try:
        return json.loads(_manifest_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_manifest(root: "os.PathLike[str] | str", manifest: Dict[str, Any]) -> None:
    """Atomically write *manifest* (tmp file + ``os.replace``)."""
    directory = extract_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(root)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmp_path, path)


def prune_manifest(root: "os.PathLike[str] | str", keep_rel: Iterable[str]) -> None:
    """Drop manifest entries for source files not in *keep_rel* (explicit-use only)."""
    manifest = load_manifest(root)
    keep = set(keep_rel)
    entries = manifest.get("entries", {})
    manifest["entries"] = {k: v for k, v in entries.items() if k in keep}
    save_manifest(root, manifest)


def needs_reextraction(root: "os.PathLike[str] | str") -> bool:
    """True if *root*'s manifest has a ``backend-missing`` stub entry and a
    backend is now installed (self-healing regeneration trigger consumed by
    ``persistence.is_stale``). False (with a single ``stat``, no read) when
    no manifest exists yet.
    """
    if not _manifest_path(root).exists():
        return False
    if not backend_available():
        return False
    manifest = load_manifest(root)
    entries = manifest.get("entries", {})
    return any(entry.get("reason") == "backend-missing" for entry in entries.values())


def _warn_backend_missing() -> None:
    global _WARNED
    if not _WARNED:
        print(
            "warning: pypdf is not installed - PDF sidecars are written as "
            "stub placeholders; install the 'pdf' extra to extract real text",
            file=sys.stderr,
        )
        _WARNED = True


def _warn_agent_only() -> None:
    global _WARNED_AGENT_ONLY
    if not _WARNED_AGENT_ONLY:
        print(
            "warning: media files were indexed as agent-transcribable stubs "
            "(no machine extractor exists for their format) - register "
            "transcripts via `retrieval sidecar --register`",
            file=sys.stderr,
        )
        _WARNED_AGENT_ONLY = True


def _warn_agent_orchestrated() -> None:
    global _WARNED_AGENT_ORCHESTRATED
    if not _WARNED_AGENT_ORCHESTRATED:
        print(
            "warning: audio/video files were indexed as agent-orchestrated "
            "stubs (run ASR via Bash, then register the transcript via "
            "`retrieval sidecar --register`)",
            file=sys.stderr,
        )
        _WARNED_AGENT_ORCHESTRATED = True


def _stub(reason: str, message: str) -> Extraction:
    return Extraction(text=message, status="stub", reason=reason, pages=0, truncated=False)


def _render_sidecar(rel: str, extraction: Extraction, *, version: str = EXTRACTOR_VERSION) -> str:
    """Render the final sidecar Markdown: a source/extractor comment header,
    then either the page transcript (``status == "ok"``) or non-empty stub
    prose naming the source file (empty sidecars yield zero chunks).

    *version* is stamped into the header's ``extractor:`` field —
    ``EXTRACTOR_VERSION`` for pypdf output, ``AGENT_EXTRACTOR_VERSION`` for
    an agent-authored transcript (see ``register_sidecar``)."""
    header = (
        f"<!-- source: {rel} -->\n"
        f"<!-- extractor: {version} status: {extraction.status} "
        f"reason: {extraction.reason} -->\n"
    )
    if extraction.status == "ok":
        return header + "\n" + extraction.text
    filename = PurePosixPath(rel).name
    body = extraction.text or "Text extraction was not possible."
    return header + f"\n# {filename}\n\n{body}\n"


def _page_too_large(page: Any) -> bool:
    """Memory guard: True if *page*'s content stream exceeds the size cap.

    ``get_contents()`` can return ``None`` (page with no content stream), so
    the ``AttributeError`` from calling ``.get_data()`` on it is caught
    explicitly rather than assuming a stream is always present.
    """
    try:
        contents = page.get_contents()
        size = len(contents.get_data()) if contents is not None else 0
    except AttributeError:
        size = 0
    return size > _PAGE_MEMORY_GUARD_BYTES


def _normalize_text(raw: str) -> str:
    """CRLF -> LF, de-hyphenate line-wrapped words, NFKC-normalize."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHEN_RE.sub(r"\1\2", text)
    return unicodedata.normalize("NFKC", text)


def _strip_headers_footers(pages_lines: List[List[str]]) -> List[List[str]]:
    """Blank out a first/last non-blank line repeated on >60% of pages,
    when the document has >=5 pages (running headers/footers)."""
    n_pages = len(pages_lines)
    if n_pages < 5:
        return pages_lines

    first_lines: List[str] = []
    last_lines: List[str] = []
    for lines in pages_lines:
        nonblank = [ln.strip() for ln in lines if ln.strip()]
        if nonblank:
            first_lines.append(nonblank[0])
            last_lines.append(nonblank[-1])

    threshold = n_pages * 0.6
    header_candidates = {ln for ln in set(first_lines) if first_lines.count(ln) > threshold}
    footer_candidates = {ln for ln in set(last_lines) if last_lines.count(ln) > threshold}
    if not header_candidates and not footer_candidates:
        return pages_lines

    result: List[List[str]] = []
    for lines in pages_lines:
        new_lines = list(lines)
        nonblank_idx = [i for i, ln in enumerate(new_lines) if ln.strip()]
        if nonblank_idx:
            first_idx = nonblank_idx[0]
            if new_lines[first_idx].strip() in header_candidates:
                new_lines[first_idx] = ""
            last_idx = nonblank_idx[-1]
            if new_lines[last_idx].strip() in footer_candidates:
                new_lines[last_idx] = ""
        result.append(new_lines)
    return result


def _split_blocks(lines: List[str]) -> List[List[str]]:
    """Split *lines* into blank-line-separated blocks of stripped, non-blank lines."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line.strip())
    if current:
        blocks.append(current)
    return blocks


def _reflow_page(lines: List[str]) -> str:
    """Reflow one page's normalized lines into joined paragraphs.

    Splits on blank lines into blocks, then within each block starts a new
    paragraph after a "short" line (a mid-block line shorter than 0.6x the
    median length of every block's non-final line — a wrapped-line-ending
    heuristic), collapses runs of spaces/tabs, and drops sub-3-char
    paragraphs (page-number/rule-line noise).
    """
    blocks = _split_blocks(lines)
    if not blocks:
        return ""

    non_final_lens = [len(ln) for block in blocks for ln in block[:-1]]
    if not non_final_lens:
        non_final_lens = [len(ln) for block in blocks for ln in block]
    median_len = statistics.median(non_final_lens) if non_final_lens else 0

    paragraphs: List[str] = []
    for block in blocks:
        current: List[str] = []
        for idx, line in enumerate(block):
            current.append(line)
            is_final_of_block = idx == len(block) - 1
            if not is_final_of_block and median_len and len(line) < 0.6 * median_len:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))

    cleaned: List[str] = []
    for para in paragraphs:
        para = _WS_COLLAPSE_RE.sub(" ", para).strip()
        if len(para) >= 3:
            cleaned.append(para)
    return "\n\n".join(cleaned)


def _truncate_at_unit_boundary(page_bodies: List[str]) -> Tuple[str, bool]:
    """Concatenate *page_bodies*, cutting at the last whole-page boundary
    that keeps the transcript under ``MAX_TRANSCRIPT_CHARS``, then append a
    truncation marker paragraph."""
    marker = "\n\n_[transcript truncated: exceeded extraction size budget]_"
    kept: List[str] = []
    total = 0
    for body in page_bodies:
        addition = ("\n\n" if kept else "") + body
        if kept and total + len(addition) + len(marker) > MAX_TRANSCRIPT_CHARS:
            break
        kept.append(body)
        total += len(addition)
    return "\n\n".join(kept) + marker, True


def _extract_pdf(data: bytes) -> Extraction:
    """Extract text from PDF *data* (pypdf backend; imported lazily)."""
    import pypdf
    from pypdf.errors import DependencyError, PyPdfError

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except DependencyError:
                return _stub("crypto-unavailable", _MSG_CRYPTO_UNAVAILABLE)
            if not decrypted:
                return _stub("encrypted", _MSG_ENCRYPTED)

        pages = reader.pages
        n_pages = len(pages)
        if n_pages == 0:
            return _stub("empty", _MSG_EMPTY)

        pages_lines: List[List[str]] = []
        thin_pages = 0
        for page in pages:
            if _page_too_large(page):
                pages_lines.append(["_[page too large to extract]_"])
                continue
            raw = page.extract_text() or ""
            if len(raw.strip()) < 100:
                thin_pages += 1
            pages_lines.append(_normalize_text(raw).split("\n"))

        if (thin_pages / n_pages) > 0.30:
            return _stub("no-text-layer", _MSG_NO_TEXT_LAYER)

        pages_lines = _strip_headers_footers(pages_lines)

        page_bodies = []
        for i, lines in enumerate(pages_lines, start=1):
            body = _reflow_page(lines)
            page_bodies.append(f"## Page {i}\n\n{body if body else '_[no text layer]_'}")

        transcript = "\n\n".join(page_bodies)
        truncated = False
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript, truncated = _truncate_at_unit_boundary(page_bodies)

        return Extraction(
            text=transcript, status="ok", reason="", pages=n_pages, truncated=truncated
        )
    except (PyPdfError, OSError, RecursionError, MemoryError, struct.error):
        return _stub("malformed", _MSG_MALFORMED)


register_extractor(".pdf", _extract_pdf)


#: Extractor-version string for ``_extract_captions`` (``.srt``/``.vtt``),
#: stamped onto its manifest entries via ``register_extractor``'s
#: ``version=`` kwarg (see ``extractor_version_for``) — distinct from
#: pypdf's ``EXTRACTOR_VERSION`` so bumping one never invalidates the
#: other's cache.
CAPTIONS_EXTRACTOR_VERSION = "captions/1"

_MSG_NO_CUES = (
    "This caption file contains no cues (timed text), so there is no "
    "transcript to extract."
)
_MSG_MALFORMED_CAPTIONS = (
    "This caption file could not be parsed as SRT or VTT (no cue-timing "
    "'-->' lines were found)."
)

#: Matches an SRT/VTT cue-timing line's two timestamps, e.g.
#: ``00:00:01,000 --> 00:00:04,000`` or ``00:04.000 --> 00:07.500``. A
#: ``search`` (not ``match``/anchor), so trailing VTT cue settings
#: (``align:middle line:90%``) after the second timestamp are ignored for
#: free — they're just text past where the regex stops looking.
_CUE_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})"
)
#: A VTT ``<v Speaker>`` or ``<v.loud Speaker>`` voice tag — captured and
#: turned into a ``Speaker: `` prefix (see ``_clean_cue_text``) before the
#: generic tag-strip below discards it like any other markup tag.
_VOICE_TAG_RE = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>")
#: Any other inline markup tag (``<b>``, ``<i>``, ``<00:00:01.500>`` karaoke
#: timestamps, etc.) — stripped with no replacement.
_TAG_RE = re.compile(r"<[^>]*>")
#: An SSA/ASS-style override tag (``{\an5}``) sometimes present in SRT
#: cues from tools that round-trip through ASS — stripped with no
#: replacement, same as an HTML-ish tag.
_ASS_OVERRIDE_RE = re.compile(r"\{\\an?\d+\}", re.IGNORECASE)

#: Every ~180s of section content opens a new ``## [HH:MM:SS] <label>``
#: heading (heading density sets chunk size — see ``_UNIT_HEADING_RE`` and
#: ``chunker.py``'s heading-change-forces-emit rule); chosen to land in the
#: 1-5 minute topical-segment range the research report recommends (§E).
_CAPTION_SECTION_SECONDS = 180.0
#: A caption paragraph closes (a new timestamp-prefixed paragraph starts)
#: once either the running text exceeds this many characters or the gap
#: since the previous cue's end exceeds 2 seconds (a natural pause).
_CAPTION_PARAGRAPH_CHARS = 400
_CAPTION_PARAGRAPH_GAP_SECONDS = 2.0


def _parse_cue_timestamp(raw: str) -> float:
    """Parse an SRT/VTT cue timestamp (``HH:MM:SS,mmm``/``HH:MM:SS.mmm``/
    ``MM:SS.mmm``) into absolute seconds."""
    parts = raw.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours = "0"
        minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _format_cue_timestamp(seconds: float) -> str:
    """Render absolute *seconds* as a zero-padded ``HH:MM:SS`` prefix,
    matching the PDF sidecar's ``## Page N`` heading slot (see the research
    report §E: sub-second precision is dropped — paragraph-level timestamps
    don't need it, and it would just be truncated by the chunker anyway)."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _decode_caption_bytes(data: bytes) -> str:
    """Decode caption *data* as UTF-8 (stripping a BOM if present), falling
    back to Latin-1 (which never raises ``UnicodeDecodeError`` — every byte
    value is a valid Latin-1 code point) for the occasional cp1252/Latin-1
    caption file in the wild. Caption extraction must never raise on bad
    encoding; a garbled-but-present transcript beats a silent crash."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _clean_cue_text(lines: List[str]) -> str:
    """Join a cue's body *lines* into one line of plain text: strip ASS
    override tags and markup tags, promote a ``<v Speaker>`` voice tag into
    a ``Speaker: `` prefix (checked before the generic tag strip, since it
    needs the tag's captured name before discarding it), and collapse
    whitespace."""
    cleaned: List[str] = []
    for line in lines:
        line = _ASS_OVERRIDE_RE.sub("", line)
        voice_match = _VOICE_TAG_RE.search(line)
        if voice_match:
            speaker = _TAG_RE.sub("", voice_match.group(1)).strip()
            line = _VOICE_TAG_RE.sub("", line)
            line = _TAG_RE.sub("", line).strip()
            line = f"{speaker}: {line}" if line else f"{speaker}:"
        else:
            line = _TAG_RE.sub("", line)
        line = _WS_COLLAPSE_RE.sub(" ", line).strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _iter_caption_cues(text: str) -> Tuple[List[Tuple[float, float, str]], bool]:
    """Tokenize *text* (already-decoded SRT or VTT content) into
    ``(start_seconds, end_seconds, cue_text)`` tuples, plus whether any
    cue-timing line was seen at all (distinguishes a malformed file from
    one that parsed fine but genuinely has zero cues).

    One tokenizer for both formats: split on blank lines into blocks (via
    the same ``_split_blocks`` the PDF reflow pipeline uses), then keep
    only blocks containing a ``-->`` line — this alone is enough to skip
    the ``WEBVTT`` header, ``NOTE``/``STYLE``/``REGION`` blocks, and a
    leading numeric SRT index line (none of those lines contain ``-->``,
    and any index/identifier line preceding the timing line within a cue's
    own block is simply not part of the body captured after it).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: List[Tuple[float, float, str]] = []
    saw_timing_line = False
    for block in _split_blocks(lines):
        timing_idx = None
        for i, line in enumerate(block):
            if "-->" in line:
                timing_idx = i
                break
        if timing_idx is None:
            continue
        match = _CUE_TIME_RE.search(block[timing_idx])
        if not match:
            continue
        saw_timing_line = True
        start = _parse_cue_timestamp(match.group(1))
        end = _parse_cue_timestamp(match.group(2))
        cue_text = _clean_cue_text(block[timing_idx + 1:])
        if not cue_text:
            continue
        cues.append((start, end, cue_text))
    return cues, saw_timing_line


def _extract_captions(data: bytes) -> Extraction:
    """Convert an SRT or VTT caption file's cues into the sidecar transcript
    format (stdlib-only, no third-party subtitle library): adjacent cues
    merge into ``[HH:MM:SS]``-prefixed paragraphs, and a new
    ``## [HH:MM:SS] <label>`` section heading opens roughly every
    ``_CAPTION_SECTION_SECONDS``. *label* is only a locator — the first ~8
    words of the section's first cue — never a real topic summary; captions
    carry no semantic labels the way an ASR-and-summarize pipeline might
    produce.
    """
    text = _decode_caption_bytes(data)
    cues, saw_timing_line = _iter_caption_cues(text)
    if not cues:
        if saw_timing_line:
            return _stub("no-cues", _MSG_NO_CUES)
        return _stub("malformed", _MSG_MALFORMED_CAPTIONS)

    # Drop consecutive-duplicate cues (identical text back-to-back, common
    # in auto-generated captions with overlapping/repeated cue windows).
    deduped: List[Tuple[float, float, str]] = []
    for cue in cues:
        if deduped and deduped[-1][2] == cue[2]:
            continue
        deduped.append(cue)

    # Merge cues into timestamp-prefixed paragraphs.
    paragraphs: List[Tuple[float, str]] = []
    para_start = para_end = None
    para_parts: List[str] = []
    for start, end, cue_text in deduped:
        if para_start is None:
            para_start, para_end, para_parts = start, end, [cue_text]
            continue
        gap = start - para_end
        joined_len = sum(len(p) for p in para_parts) + len(cue_text)
        if gap > _CAPTION_PARAGRAPH_GAP_SECONDS or joined_len > _CAPTION_PARAGRAPH_CHARS:
            paragraphs.append((para_start, " ".join(para_parts)))
            para_start, para_end, para_parts = start, end, [cue_text]
        else:
            para_parts.append(cue_text)
            para_end = end
    if para_parts:
        paragraphs.append((para_start, " ".join(para_parts)))

    # Group paragraphs into ~180s sections; always at least one.
    sections: List[List[Tuple[float, str]]] = []
    section_opened_at = None
    for start, para_text in paragraphs:
        if section_opened_at is None or (start - section_opened_at) >= _CAPTION_SECTION_SECONDS:
            sections.append([])
            section_opened_at = start
        sections[-1].append((start, para_text))

    section_bodies = []
    for section in sections:
        first_start, first_text = section[0]
        label = " ".join(first_text.split()[:8])
        heading = f"## [{_format_cue_timestamp(first_start)}] {label}"
        body = "\n\n".join(
            f"[{_format_cue_timestamp(start)}] {para_text}" for start, para_text in section
        )
        section_bodies.append(f"{heading}\n\n{body}")

    transcript = "\n\n".join(section_bodies)
    truncated = False
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript, truncated = _truncate_at_unit_boundary(section_bodies)

    return Extraction(
        text=transcript, status="ok", reason="", pages=len(sections), truncated=truncated
    )


register_extractor(".srt", _extract_captions, version=CAPTIONS_EXTRACTOR_VERSION)
register_extractor(".vtt", _extract_captions, version=CAPTIONS_EXTRACTOR_VERSION)


def _engine_version() -> str:
    """Lazy import of ``retrieval.__version__`` (avoids a module-scope
    circular import: ``retrieval/__init__.py`` imports ``persistence``,
    which imports ``project_loader``, which imports this module)."""
    try:
        from retrieval import __version__
    except ImportError:  # pragma: no cover - defensive fallback
        return "unknown"
    return __version__


def _is_cache_hit(
    entry: Dict[str, Any] | None, sha256: str, sidecar_path: Path, force: bool
) -> bool:
    if force or entry is None:
        return False
    if entry.get("sha256") != sha256:
        return False
    if entry.get("extractor_version") not in _ACCEPTED_EXTRACTOR_VERSIONS:
        return False
    # A hand-corrupted manifest entry missing either key is treated as a
    # cache miss (triggering a fresh extraction), not a KeyError later in
    # ensure_sidecar's cache-hit branch (which reads entry["status"]).
    if not entry.get("status"):
        return False
    if not entry.get("sidecar"):
        return False
    try:
        actual_size = sidecar_path.stat().st_size
    except OSError:
        return False
    if actual_size != entry.get("sidecar_bytes"):
        return False
    if entry.get("reason") == "backend-missing" and backend_available():
        return False
    return True


#: Manifest ``entries[rel]["identity"]`` value for the common case: the
#: sibling ``sha256`` field is a real SHA-256 content hash of the source
#: file's bytes.
IDENTITY_SHA256 = "sha256/1"

#: Manifest ``entries[rel]["identity"]`` value for
#: ``AGENT_ORCHESTRATED_EXTENSIONS`` (tier-3 audio/video) only: the sibling
#: ``sha256`` field is *not* a content hash — it's
#: ``sha256("stat/1|{st_size}|{st_mtime_ns}")``, computed from a single
#: ``os.stat()`` call, never a byte read. Stored under the existing
#: ``sha256`` key (not a new key) so ``_is_cache_hit``'s comparison logic
#: needs no change; ``identity`` is purely informational, distinguishing
#: "this looks like a hash but isn't one" for a reader of the manifest.
IDENTITY_STAT = "stat/1"


def _source_identity(
    path: Path, *, stat: "os.stat_result | None" = None, data: bytes | None = None
) -> Tuple[str, int, str]:
    """Return ``(digest, source_bytes, identity)`` for *path* — the single
    helper every read of a source file's identity goes through (replacing
    three formerly-independent ``read_bytes()`` call sites in
    ``ensure_sidecar``, ``register_sidecar``, and ``_entry_state``).

    For every tier except ``AGENT_ORCHESTRATED_EXTENSIONS`` (tier-3
    audio/video), *digest* is a real SHA-256 of the file's bytes (identity
    ``IDENTITY_SHA256``) — *data*, when given, reuses bytes the caller
    already read (``ensure_sidecar`` needs the raw bytes for extraction
    anyway; passing them here avoids a second read) rather than re-reading.

    For tier-3, *digest* is instead ``sha256("stat/1|{st_size}|{st_mtime_ns}")``
    (identity ``IDENTITY_STAT``) computed from an ``os.stat()`` call ALONE —
    this function never calls ``path.read_bytes()`` for a tier-3 suffix,
    full stop, matching ``AGENT_ORCHESTRATED_EXTENSIONS``'s defining
    property that the engine never reads these files' bytes. *stat*, when
    given, reuses the caller's own already-taken ``os.stat()`` result (see
    ``ensure_sidecar``, which stats *source_path* early for its process-memo
    key) instead of calling it again — one syscall total, and the same
    read-then-use-that-same-read TOCTOU discipline the byte-hash path has.

    Trade-off (tier-3 only, documented per the research report's Open
    Question): an in-place edit that happens to preserve both file size and
    mtime produces a false cache hit under ``"stat/1"`` — impossible under
    ``"sha256/1"``, which would catch any content change. ``retrieval
    extract --force`` / ``ensure_sidecar(..., force=True)`` is the escape
    hatch when that matters (e.g. after editing a video file in place with a
    tool that preserves its mtime).
    """
    if path.suffix.lower() in AGENT_ORCHESTRATED_EXTENSIONS:
        st = stat if stat is not None else path.stat()
        digest = hashlib.sha256(
            f"stat/1|{st.st_size}|{st.st_mtime_ns}".encode("ascii")
        ).hexdigest()
        return digest, st.st_size, IDENTITY_STAT
    if data is None:
        data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), IDENTITY_SHA256


def ensure_sidecar(
    root: "os.PathLike[str] | str", source: "os.PathLike[str] | str", *, force: bool = False
) -> Sidecar:
    """Return the sidecar for *source* (a file under *root*), extracting
    (and caching) it if needed.

    Reads *source*'s bytes exactly once (hashes and extracts from the same
    read — avoids a TOCTOU window between "hash it" and "extract it"),
    checked first against a process-level memo keyed by
    ``(abs_path, st_size, st_mtime_ns)`` (cheap re-calls within one
    ``index --auto`` run), then against the on-disk manifest (cache hit
    requires: entry present, sha256 match, extractor-version match, sidecar
    file present with matching size, and not a stale ``backend-missing``
    stub now that a backend is installed).
    """
    root_path = Path(root).resolve()
    source_path = Path(source).resolve()
    rel = source_path.relative_to(root_path).as_posix()
    stat = source_path.stat()
    memo_key = (str(source_path), stat.st_size, stat.st_mtime_ns)
    if not force and memo_key in _PROCESS_MEMO:
        return _PROCESS_MEMO[memo_key]

    sidecar_rel = sidecar_relpath(rel)
    sidecar_path = root_path / sidecar_rel

    manifest = load_manifest(root_path)
    entries = manifest.setdefault("entries", {})
    entry = entries.get(rel)

    suffix = source_path.suffix.lower()
    is_tier3 = suffix in AGENT_ORCHESTRATED_EXTENSIONS
    # Tier-3 (audio/video): never read the source's bytes — _source_identity
    # stat-hashes it instead. `data` stays empty and is never handed to an
    # extractor below (tier-3 has none registered, so that branch is never
    # taken for these suffixes).
    data = b"" if is_tier3 else source_path.read_bytes()
    sha256, source_bytes, identity = _source_identity(
        source_path, stat=stat, data=None if is_tier3 else data
    )

    if _is_cache_hit(entry, sha256, sidecar_path, force):
        text = sidecar_path.read_text(encoding="utf-8")
        sidecar = Sidecar(
            docid=sidecar_rel, path=sidecar_path, text=text,
            status=entry.get("status", "ok"), reason=entry.get("reason", ""), cache_hit=True,
        )
        _PROCESS_MEMO[memo_key] = sidecar
        return sidecar

    extractor = extractor_for(source_path)
    if extractor is None:
        # No machine extractor exists at all for this suffix (agent-only
        # media, or agent-orchestrated tier-3 audio/video): never a
        # "backend missing" situation and must never look like one —
        # needs_reextraction/_is_cache_hit key on the literal
        # "backend-missing" reason to self-heal a stub once pypdf becomes
        # available, and a no-extractor stub must never be caught by that
        # (it would loop forever re-stubbing a docx or an mp4). The two
        # no-extractor tiers get distinct messages/reasons (never
        # "backend-missing" for either) — see _warn_agent_only /
        # _warn_agent_orchestrated and the anti-loop invariant above.
        if is_tier3:
            _warn_agent_orchestrated()
            extraction = _stub("agent-orchestrated", _MSG_AGENT_ORCHESTRATED)
        else:
            _warn_agent_only()
            extraction = _stub("agent-only", _MSG_AGENT_ONLY)
    elif suffix in PYPDF_EXTENSIONS and not backend_available():
        # A registered extractor exists but needs the pypdf backend, which
        # isn't installed — distinct from the no-extractor branch above:
        # this one DOES self-heal once pypdf becomes available.
        _warn_backend_missing()
        extraction = _stub("backend-missing", _MSG_BACKEND_MISSING)
    else:
        extraction = extractor(data)

    sidecar_text = _render_sidecar(rel, extraction, version=extractor_version_for(source_path))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar_text, encoding="utf-8")

    entries[rel] = {
        "sha256": sha256,
        "source_bytes": source_bytes,
        "identity": identity,
        "extractor_version": extractor_version_for(source_path),
        "sidecar": sidecar_rel,
        "sidecar_bytes": len(sidecar_text.encode("utf-8")),
        "pages": extraction.pages,
        "status": extraction.status,
        "reason": extraction.reason,
        "truncated": extraction.truncated,
    }
    manifest["schema"] = 1
    manifest["extractor_version"] = EXTRACTOR_VERSION
    manifest["engine_version"] = _engine_version()
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_manifest(root_path, manifest)

    sidecar = Sidecar(
        docid=sidecar_rel, path=sidecar_path, text=sidecar_text,
        status=extraction.status, reason=extraction.reason, cache_hit=False,
    )
    _PROCESS_MEMO[memo_key] = sidecar
    return sidecar


def _truncate_agent_transcript(text: str) -> Tuple[str, bool]:
    """Truncate an over-budget agent-authored *text* to fit
    ``MAX_TRANSCRIPT_CHARS``, preferring a whole-unit-boundary cut (reusing
    ``_truncate_at_unit_boundary``) when ``## Page N`` or ``## [HH:MM:SS] ...``
    headings are present, else a plain character slice with the same
    truncation marker."""
    heading_starts = [m.start() for m in _UNIT_HEADING_RE.finditer(text)]
    if heading_starts:
        segments = []
        for i, start in enumerate(heading_starts):
            end = heading_starts[i + 1] if i + 1 < len(heading_starts) else len(text)
            segments.append(text[start:end].rstrip("\n"))
        return _truncate_at_unit_boundary(segments)
    marker = "\n\n_[transcript truncated: exceeded extraction size budget]_"
    return text[: MAX_TRANSCRIPT_CHARS - len(marker)] + marker, True


def register_sidecar(
    root: "os.PathLike[str] | str",
    source: "os.PathLike[str] | str",
    transcript: str,
    *,
    truncated: bool = False,
) -> Sidecar:
    """Register an agent-authored *transcript* as *source*'s sidecar.

    The no-pypdf recovery path for PDFs, and the **only** indexing path for
    ``AGENT_ONLY_EXTENSIONS`` media (docx/pptx/xlsx/images), which have no
    machine extractor at all: a coding agent that read *source* natively
    (e.g. via its own ``Read`` tool, or vision for an image) can hand-write
    the extracted text and register it here instead of (or in place of)
    installing the ``pdf`` extra. Never imports ``pypdf`` on any path — this
    function works identically whether or not the backend is installed.

    Writes the same sidecar-Markdown shape ``ensure_sidecar`` would (source
    + extractor header, then the transcript), stamped with
    ``AGENT_EXTRACTOR_VERSION`` so ``_is_cache_hit`` recognizes it as fresh
    and ``needs_reextraction``/``extract --force`` know not to silently
    treat it as a stale pypdf stub (see module docstring's supersede
    policy). *truncated* is the caller-declared truncation flag when the
    caller already truncated its own transcript upstream; this function
    additionally truncates at ``MAX_TRANSCRIPT_CHARS`` itself if needed,
    OR-ing the two flags together.
    """
    root_path = Path(root).resolve()
    source_path = Path(source).resolve()
    try:
        rel = source_path.relative_to(root_path).as_posix()
    except ValueError:
        raise ValueError(f"{source_path} is not under root {root_path}") from None
    if not source_path.is_file():
        raise ValueError(f"source file not found: {source_path}")
    if source_path.suffix.lower() not in EXTRACTABLE_EXTENSIONS:
        raise ValueError(
            f"{rel!r} has a suffix not in EXTRACTABLE_EXTENSIONS "
            f"({sorted(EXTRACTABLE_EXTENSIONS)})"
        )

    if not transcript.strip():
        raise ValueError("transcript is empty")

    text = _LEADING_COMMENT_RE.sub("", transcript)
    pages = len(_UNIT_HEADING_RE.findall(text))
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text, was_truncated = _truncate_agent_transcript(text)
        truncated = truncated or was_truncated

    # Never read_bytes() a tier-3 (audio/video) source — _source_identity
    # stat-hashes it instead, same as ensure_sidecar.
    sha256, source_bytes, identity = _source_identity(source_path)

    extraction = Extraction(text=text, status="ok", reason="", pages=pages, truncated=truncated)
    sidecar_text = _render_sidecar(rel, extraction, version=AGENT_EXTRACTOR_VERSION)
    sidecar_rel = sidecar_relpath(rel)
    sidecar_path = root_path / sidecar_rel
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar_text, encoding="utf-8")
    sidecar_sha256 = hashlib.sha256(sidecar_text.encode("utf-8")).hexdigest()

    manifest = load_manifest(root_path)
    entries = manifest.setdefault("entries", {})
    entries[rel] = {
        "sha256": sha256,
        "source_bytes": source_bytes,
        "identity": identity,
        "extractor_version": AGENT_EXTRACTOR_VERSION,
        "sidecar": sidecar_rel,
        "sidecar_bytes": len(sidecar_text.encode("utf-8")),
        "pages": pages,
        "status": "ok",
        "reason": "",
        "truncated": truncated,
        "authored_by": AUTHORED_BY_AGENT,
        "authored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sidecar_sha256": sidecar_sha256,
    }
    manifest["schema"] = 1
    manifest["extractor_version"] = EXTRACTOR_VERSION
    manifest["engine_version"] = _engine_version()
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_manifest(root_path, manifest)

    invalidate_process_cache(source_path)

    return Sidecar(
        docid=sidecar_rel, path=sidecar_path, text=sidecar_text,
        status="ok", reason="", cache_hit=False,
    )


def agent_sidecar_revision(entry: Dict[str, Any]) -> str:
    """Return *entry*'s ``sidecar_sha256`` when it's an agent-authored
    manifest entry, else ``""``.

    Feeds ``persistence.compute_fingerprint``: mixing this into a
    discovered file's fingerprint line makes an agent transcript's own
    content (not just the source PDF's stat) part of staleness detection,
    so re-registering a changed transcript triggers a reindex even though
    the source PDF's own bytes/mtime never changed.
    """
    if entry.get("authored_by") != AUTHORED_BY_AGENT:
        return ""
    return entry.get("sidecar_sha256", "")


def _entry_state(entry: Dict[str, Any] | None, source_path: Path) -> str:
    """Classify one source file's manifest *entry* into a ``sidecar_states``
    state label: ``missing`` (no entry yet), ``outdated`` (entry's
    ``sha256`` no longer matches the source file's current content),
    ``agent-authored`` (written by ``register_sidecar``, source unchanged),
    ``stub`` (entry's ``status`` is ``"stub"``, e.g. a ``backend-missing``
    placeholder), or ``ok`` (a fresh pypdf-extracted transcript)."""
    if entry is None:
        return "missing"
    # Never read_bytes() a tier-3 (audio/video) source (see
    # AGENT_ORCHESTRATED_EXTENSIONS) — `retrieval sidecar --list` must stay
    # a stat-only operation for these, same as ensure_sidecar/register_sidecar.
    sha256, _source_bytes, _identity = _source_identity(source_path)
    if entry.get("sha256") != sha256:
        return "outdated"
    if entry.get("authored_by") == AUTHORED_BY_AGENT:
        return "agent-authored"
    if entry.get("status") == "stub":
        return "stub"
    return "ok"


def _sidecar_state_record(rel: str, entry: Dict[str, Any] | None, state: str) -> Dict[str, Any]:
    """One ``sidecar_states`` result record for *rel*'s *entry* (``None``
    when *state* is ``"missing"``) and its classified *state*."""
    return {
        "rel": rel,
        "state": state,
        "sidecar": entry.get("sidecar") if entry else None,
        "status": entry.get("status") if entry else None,
        "reason": entry.get("reason", "") if entry else "",
        "pages": entry.get("pages", 0) if entry else 0,
        "authored_at": entry.get("authored_at") if entry else None,
    }


def sidecar_states(
    root: "os.PathLike[str] | str", sources: Iterable["os.PathLike[str] | str"]
) -> List[Dict[str, Any]]:
    """Report each of *sources*' sidecar state, for ``retrieval sidecar
    --list`` (see ``_entry_state`` for the state taxonomy)."""
    root_path = Path(root).resolve()
    entries = load_manifest(root_path).get("entries", {})
    results: List[Dict[str, Any]] = []
    for source in sources:
        source_path = Path(source).resolve()
        rel = source_path.relative_to(root_path).as_posix()
        entry = entries.get(rel)
        results.append(_sidecar_state_record(rel, entry, _entry_state(entry, source_path)))
    return results
