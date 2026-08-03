"""Sidecar-transcript extraction for non-text-native document formats (PDF).

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

#: File suffixes this module can produce a sidecar for.
EXTRACTABLE_EXTENSIONS = frozenset({".pdf"})

#: Raw source-file byte cap applied at discovery time for extractable
#: suffixes (see ``project_loader.discover_files``'s ``extract_max_bytes``).
EXTRACT_MAX_BYTES = 25_000_000

#: Post-extraction transcript character budget; a longer transcript is
#: truncated at the last whole-page boundary under this limit (see
#: ``_truncate_at_page_boundary``).
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
    "extract real content."
)


def register_extractor(suffix: str, fn: Callable[[bytes], Extraction]) -> None:
    """Register *fn* as the extractor for files with *suffix* (e.g. ``.pdf``)."""
    _EXTRACTORS[suffix.lower()] = fn


def extractor_for(path: "os.PathLike[str] | str") -> Callable[[bytes], Extraction] | None:
    """Return the registered extractor for *path*'s suffix, or ``None``."""
    return _EXTRACTORS.get(Path(path).suffix.lower())


def require_extractors(extensions: Iterable[str]) -> None:
    """Raise a guidance ``RuntimeError`` if *extensions* needs a backend that
    isn't installed (mirrors ``retrieval.retrievers``' guidance-``RuntimeError``
    convention: ``RuntimeError``, not ``ImportError``, two-line message).

    Only checks extension membership in ``EXTRACTABLE_EXTENSIONS`` — never
    probes import success for eligibility/discovery, only for this explicit
    preflight call.
    """
    if not any(ext.lower() in EXTRACTABLE_EXTENSIONS for ext in extensions):
        return
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - guidance path
        raise RuntimeError(
            "PDF extraction needs the 'pdf' extra:\n"
            "  uv pip install -e '.[pdf]'"
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


def _stub(reason: str, message: str) -> Extraction:
    return Extraction(text=message, status="stub", reason=reason, pages=0, truncated=False)


def _render_sidecar(rel: str, extraction: Extraction) -> str:
    """Render the final sidecar Markdown: a source/extractor comment header,
    then either the page transcript (``status == "ok"``) or non-empty stub
    prose naming the source file (empty sidecars yield zero chunks)."""
    header = (
        f"<!-- source: {rel} -->\n"
        f"<!-- extractor: {EXTRACTOR_VERSION} status: {extraction.status} "
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


def _truncate_at_page_boundary(page_bodies: List[str]) -> Tuple[str, bool]:
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
            transcript, truncated = _truncate_at_page_boundary(page_bodies)

        return Extraction(
            text=transcript, status="ok", reason="", pages=n_pages, truncated=truncated
        )
    except (PyPdfError, OSError, RecursionError, MemoryError, struct.error):
        return _stub("malformed", _MSG_MALFORMED)


register_extractor(".pdf", _extract_pdf)


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
    if entry.get("extractor_version") != EXTRACTOR_VERSION:
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

    data = source_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()

    if _is_cache_hit(entry, sha256, sidecar_path, force):
        text = sidecar_path.read_text(encoding="utf-8")
        sidecar = Sidecar(
            docid=sidecar_rel, path=sidecar_path, text=text,
            status=entry["status"], reason=entry.get("reason", ""), cache_hit=True,
        )
        _PROCESS_MEMO[memo_key] = sidecar
        return sidecar

    if backend_available():
        extractor = extractor_for(source_path) or _extract_pdf
        extraction = extractor(data)
    else:
        _warn_backend_missing()
        extraction = _stub("backend-missing", _MSG_BACKEND_MISSING)

    sidecar_text = _render_sidecar(rel, extraction)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar_text, encoding="utf-8")

    entries[rel] = {
        "sha256": sha256,
        "source_bytes": len(data),
        "extractor_version": EXTRACTOR_VERSION,
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
