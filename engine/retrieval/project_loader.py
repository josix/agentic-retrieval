"""Load documents/chunks from an invoking project's directory tree.

Stdlib-only (``os``, ``pathlib``, ``fnmatch``) — no network libraries — so the
default retrieval pipeline stays fully offline. Discovers text-like files
under a project root, skipping VCS/dependency/build directories and files
that look like secrets by name.

A file is eligible for discovery if its suffix is in ``DEFAULT_EXTENSIONS``
*or* its basename case-insensitively matches ``DEFAULT_INCLUDE_BASENAMES``
(well-known extensionless project files such as ``Dockerfile``, ``Makefile``,
``LICENSE``); either way it must still pass the exclude-glob and size checks.

The filename deny-list (``DEFAULT_EXCLUDE_GLOBS``) is a **best-effort**
guard against accidentally indexing credential files by name — it is not
content scanning. A file named ``notes.txt`` containing an API key will
still be indexed; do not rely on this module for secret detection.
"""

import fnmatch
import os
from pathlib import Path
from typing import Iterable, List, Optional

from retrieval.ast_chunker import chunk_code, language_for_path
from retrieval.chunker import Chunk, chunk_document
from retrieval.document import Document

DEFAULT_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".rst",
        ".markdown",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".cs",
        ".sh",
        ".bash",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".json",
        ".html",
        ".css",
        ".scss",
        ".sql",
        ".kt",
        ".swift",
        ".scala",
        ".r",
        ".lua",
        ".pl",
    }
)

DEFAULT_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".claude",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "target",
        "site",
        ".next",
        ".cache",
        "site-packages",
        ".complexipy_cache",
        ".agentic-retrieval",
    }
)

DEFAULT_INCLUDE_BASENAMES = frozenset(
    {
        "dockerfile",
        "makefile",
        "license",
        "licence",
        "readme",
        "changelog",
        "notice",
        "vagrantfile",
        "jenkinsfile",
        "procfile",
    }
)

DEFAULT_EXCLUDE_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.p8",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*.keystore",
    "*.jks",
    "*.htpasswd",
    ".npmrc",
    ".pypirc",
    "*credentials*",
    "*secret*",
)

MAX_FILE_BYTES = 1_000_000


def _is_excluded_dir(dirname: str, exclude_dirs: frozenset) -> bool:
    """Return True if dirname should be pruned from traversal."""
    return dirname in exclude_dirs or dirname.endswith(".egg-info")


def _matches_any_glob(filename: str, exclude_globs: Iterable[str]) -> bool:
    """Case-insensitive fnmatch of filename against each glob pattern."""
    lowered = filename.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in exclude_globs)


def _has_allowed_name(
    filename: str, extensions: frozenset, include_basenames: frozenset
) -> bool:
    """Return True if *filename*'s suffix or basename is on an allowlist."""
    suffix = Path(filename).suffix.lower()
    if suffix in extensions:
        return True
    return filename.lower() in include_basenames


def _is_eligible_file(
    file_path: Path,
    extensions: frozenset,
    include_basenames: frozenset,
    exclude_globs: Iterable[str],
    max_bytes: int,
) -> bool:
    """Return True if *file_path* passes the name allowlist, deny-list, and size checks."""
    filename = file_path.name
    if not _has_allowed_name(filename, extensions, include_basenames):
        return False
    if _matches_any_glob(filename, exclude_globs):
        return False
    try:
        return file_path.stat().st_size <= max_bytes
    except OSError:
        return False


def discover_files(
    root: "os.PathLike[str] | str",
    *,
    extensions: frozenset = DEFAULT_EXTENSIONS,
    exclude_dirs: frozenset = DEFAULT_EXCLUDE_DIRS,
    exclude_globs: Iterable[str] = DEFAULT_EXCLUDE_GLOBS,
    include_basenames: frozenset = DEFAULT_INCLUDE_BASENAMES,
    max_bytes: int = MAX_FILE_BYTES,
) -> List[Path]:
    """Walk *root* and return sorted, deterministic list of eligible file paths.

    Prunes ``exclude_dirs`` (and any dir ending in ``.egg-info``) in-place
    during ``os.walk`` so excluded subtrees are never descended into. A file
    is eligible if its extension is in *extensions* or its basename
    case-insensitively matches *include_basenames*, and it also passes the
    filename deny-list and size checks. Symlinks are not followed.
    """
    root_path = Path(root)
    found: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d, exclude_dirs)]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if _is_eligible_file(
                file_path, extensions, include_basenames, exclude_globs, max_bytes
            ):
                found.append(file_path)

    return sorted(found)


def read_text_safe(path: "os.PathLike[str] | str") -> Optional[str]:
    """Read *path* as UTF-8 text, returning None if it looks binary or unreadable.

    Returns None when the first 1024 bytes contain a null byte, the content
    fails UTF-8 decoding, or the file cannot be opened/read.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(1024)
        if b"\x00" in head:
            return None
        with open(path, "rb") as fh:
            raw = fh.read()
        return raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def load_documents(
    root: "os.PathLike[str] | str",
    **kw,
) -> List[Document]:
    """Discover files under *root* and return one Document per readable file.

    ``docid`` is the file's POSIX-style path relative to *root*; ``url`` is
    left blank (no network association). Files that fail to decode as text
    are skipped.
    """
    root_path = Path(root)
    documents: List[Document] = []

    for file_path in discover_files(root_path, **kw):
        text = read_text_safe(file_path)
        if text is None:
            continue
        docid = file_path.relative_to(root_path).as_posix()
        documents.append(Document(docid=docid, text=text, url=""))

    return documents


def load_chunks(
    root: "os.PathLike[str] | str",
    **kw,
) -> List[Chunk]:
    """Discover files under *root*, chunk each one, and return a flat list.

    Each file's relative POSIX path is used as ``doc_id`` when chunking, so
    chunk ids and doc_ids trace back to a real project path.
    """
    chunks: List[Chunk] = []
    for document in load_documents(root, **kw):
        chunks.extend(chunk_document(document.docid, document.text))
    return chunks


def load_chunk_documents(
    root: "os.PathLike[str] | str",
    **kw,
) -> List[Document]:
    """Discover files under *root*, chunk each one, and return one chunk-
    granularity ``Document`` per chunk.

    Each returned ``Document``'s ``docid`` is ``"{path}:{start}-{end}"``
    (the file's relative POSIX path plus its 1-based line span), with
    ``source_path``/``start_line``/``end_line`` set from the chunk's span —
    this is what the production retrievers (see ``retrieval/retrievers.py``)
    index so search results can point a coding agent at an exact
    ``file:line`` location. A file with only blank content yields no chunks
    and therefore no Documents.
    """
    documents: List[Document] = []
    for document in load_documents(root, **kw):
        for chunk in chunk_document(document.docid, document.text):
            docid = f"{document.docid}:{chunk.start_line}-{chunk.end_line}"
            documents.append(
                Document(
                    docid=docid,
                    text=chunk.text,
                    url=document.url,
                    source_path=document.docid,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
            )
    return documents


def load_ast_chunk_documents(
    root: "os.PathLike[str] | str",
    **kw,
) -> List[Document]:
    """Discover files under *root* and return one AST-boundary chunk-
    granularity ``Document`` per chunk (see ``retrieval.ast_chunker``).

    For each file, resolves a tree-sitter language from its suffix (via
    ``language_for_path``); when a language is found, chunks with
    ``chunk_code`` at AST node boundaries, carrying a breadcrumb
    ``context`` (enclosing function/class path, e.g. ``"Bar.baz"``) on each
    Document. Files with no mapped language, or whose AST chunking returns
    no chunks (e.g. a severely broken parse), fall back to the line-based
    ``chunk_document`` — same as ``load_chunk_documents`` — with an empty
    ``context``.

    A missing ``tree-sitter-language-pack`` install surfaces as a
    ``RuntimeError`` (raised by ``chunk_code``) and is *not* caught here: it
    is the signal an all-mode index run uses to skip this strategy (see
    ``retrieval.cli._index_all``).
    """
    documents: List[Document] = []
    for document in load_documents(root, **kw):
        language = language_for_path(document.docid)
        chunks = chunk_code(document.docid, document.text, language) if language else []
        if not chunks:
            chunks = chunk_document(document.docid, document.text)
        for chunk in chunks:
            docid = f"{document.docid}:{chunk.start_line}-{chunk.end_line}"
            documents.append(
                Document(
                    docid=docid,
                    text=chunk.text,
                    url=document.url,
                    source_path=document.docid,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    context=getattr(chunk, "context", ""),
                )
            )
    return documents
