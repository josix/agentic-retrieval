"""Per-entry loader for gettext ``.po`` files (translation-memory indexing).

Stdlib-only (``re``, ``dataclasses``, ``pathlib``) — no ``polib``. A lenient
line parser turns each live PO entry into one chunk-granularity ``Document``
so a translator can search msgid/msgstr pairs and ``Read`` the exact entry.

Policy:

* The header entry (``msgid ""`` without ``msgctxt``) is never indexed.
* Obsolete entries (``#~`` lines) are never indexed.
* Fuzzy and untranslated entries *are* indexed but flagged in ``metadata`` so
  callers can filter.
* A ``Document``'s span (``start_line``..``end_line``) covers the whole entry
  block, translator comments included. ``context`` stays empty on purpose so
  the tree-sitter retriever does not prefix it into the ranked text.

The parser never raises on malformed input; unrecognised lines are skipped.

This module registers itself as the ``.po`` record loader (see
``retrieval.record_loaders``) and is listed in
``record_loaders.BUILTIN_LOADER_MODULES``.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from retrieval import record_loaders
from retrieval.document import Document

PO_KIND = "po_entry"
PO_SUFFIXES = frozenset({".po"})

_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    '"': '"',
    "\\": "\\",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
}
_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
_KEYWORD_RE = re.compile(r"^(msgctxt|msgid_plural|msgid|msgstr)(?:\[(\d+)\])?\s*(.*)$")


def is_po_path(path: str) -> bool:
    """True if *path* has a ``.po`` suffix (case-insensitive).

    Convenience predicate; the engine itself routes via ``record_loaders.lookup``.
    """
    return Path(path).suffix.lower() in PO_SUFFIXES


def _ends_with_escaped_quote(body: str) -> bool:
    """True if the trailing quote of *body* is preceded by an odd backslash run."""
    before = body[:-1]
    return (len(before) - len(before.rstrip("\\"))) % 2 == 1


def unescape_po_string(quoted_body: str) -> str:
    """Decode a quoted PO string literal (``"..."``) into its text.

    Uses an explicit escape map rather than ``unicode_escape``, which would
    corrupt non-ASCII (Chinese) text. Unknown escapes keep the backslash.
    """
    body = quoted_body.strip()
    if body.startswith('"'):
        body = body[1:]
        if body.endswith('"') and not _ends_with_escaped_quote(body):
            body = body[:-1]
    return _ESCAPE_RE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(0)), body)


@dataclass
class PoEntry:
    msgid: str
    msgstr: str  # plural forms joined with "\n"
    start_line: int
    end_line: int
    msgid_line: int
    msgctxt: Optional[str] = None
    msgid_plural: Optional[str] = None
    msgstr_plural: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    @property
    def fuzzy(self) -> bool:
        return "fuzzy" in self.flags

    @property
    def is_header(self) -> bool:
        return self.msgid == "" and self.msgctxt is None

    @property
    def translated(self) -> bool:
        if self.msgstr_plural:
            return all(self.msgstr_plural)
        return self.msgstr != ""


class _Block:
    """Mutable accumulator for the entry currently being parsed."""

    def __init__(self) -> None:
        self.start: Optional[int] = None
        self.end = 0
        self.msgid_line = 0
        self.fields: Dict[str, str] = {}
        self.plural: Dict[int, str] = {}
        self.flags: List[str] = []
        self.references: List[str] = []
        self.target: Optional[str] = None  # field name, or "msgstr[N]"
        self.has_msgstr = False

    def touch(self, lineno: int) -> None:
        if self.start is None:
            self.start = lineno
        self.end = max(self.end, lineno)

    def append(self, name: str, text: str) -> None:
        if name.startswith("msgstr["):
            index = int(name[7:-1])
            self.plural[index] = self.plural.get(index, "") + text
        else:
            self.fields[name] = self.fields.get(name, "") + text

    def to_entry(self) -> Optional[PoEntry]:
        if "msgid" not in self.fields or self.start is None:
            return None
        plural = [self.plural[i] for i in sorted(self.plural)]
        msgstr = "\n".join(plural) if plural else self.fields.get("msgstr", "")
        return PoEntry(
            msgid=self.fields["msgid"],
            msgstr=msgstr,
            start_line=self.start,
            end_line=self.end,
            msgid_line=self.msgid_line,
            msgctxt=self.fields.get("msgctxt"),
            msgid_plural=self.fields.get("msgid_plural"),
            msgstr_plural=plural,
            flags=self.flags,
            references=self.references,
        )


def _apply_comment(block: _Block, line: str) -> None:
    if line.startswith("#:"):
        block.references.extend(line[2:].split())
    elif line.startswith("#,"):
        block.flags.extend(f.strip() for f in line[2:].split(",") if f.strip())


def _apply_keyword(block: _Block, match: "re.Match[str]", lineno: int) -> None:
    keyword, index, rest = match.group(1), match.group(2), match.group(3)
    name = f"msgstr[{int(index)}]" if keyword == "msgstr" and index is not None else keyword
    block.target = name
    if keyword == "msgid":
        block.msgid_line = lineno
    if keyword == "msgstr":
        block.has_msgstr = True
    block.append(name, unescape_po_string(rest) if rest else "")


def _strip_bom(lines: List[str]) -> List[str]:
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0][1:]
    return lines


def _handle_line(block: _Block, line: str, lineno: int) -> bool:
    """Apply one non-blank line to *block*; True if *block* must be flushed first."""
    if line.startswith("#~"):
        return False
    if line.startswith("#"):
        flush_first = block.has_msgstr
        if flush_first:
            return True
        block.touch(lineno)
        _apply_comment(block, line)
        return False
    if line.startswith('"'):
        if block.target is not None:
            block.append(block.target, unescape_po_string(line))
            block.touch(lineno)
        return False
    match = _KEYWORD_RE.match(line)
    if match is None:
        return False
    if match.group(1) in ("msgctxt", "msgid") and block.has_msgstr:
        return True
    block.touch(lineno)
    _apply_keyword(block, match, lineno)
    return False


def parse_po(text: str) -> List[PoEntry]:
    """Parse PO *text* into live entries (header included, obsolete excluded)."""
    entries: List[PoEntry] = []
    block = _Block()

    def flush() -> None:
        nonlocal block
        entry = block.to_entry()
        if entry is not None:
            entries.append(entry)
        block = _Block()

    for lineno, raw in enumerate(_strip_bom(text.splitlines()), start=1):
        line = raw.strip()
        if not line:
            flush()
        elif _handle_line(block, line, lineno):
            flush()
            _handle_line(block, line, lineno)
    flush()
    return entries


def _entry_metadata(entry: PoEntry) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "kind": PO_KIND,
        "msgid": entry.msgid,
        "msgstr": entry.msgstr,
        "fuzzy": entry.fuzzy,
        "translated": entry.translated,
        "line": entry.msgid_line,
        "references": list(entry.references),
    }
    if entry.msgctxt is not None:
        metadata["msgctxt"] = entry.msgctxt
    if entry.msgid_plural is not None:
        metadata["msgid_plural"] = entry.msgid_plural
        metadata["msgstr_plural"] = list(entry.msgstr_plural)
    return metadata


def po_entries_to_documents(rel_path: str, text: str, url: str = "") -> List[Document]:
    """One ``Document`` per live, non-header entry of the PO file *text*."""
    documents: List[Document] = []
    for entry in parse_po(text):
        if entry.is_header:
            continue
        parts = (entry.msgctxt, entry.msgid, entry.msgid_plural, entry.msgstr)
        documents.append(
            Document(
                docid=f"{rel_path}:{entry.start_line}-{entry.end_line}",
                text="\n".join(p for p in parts if p),
                url=url,
                source_path=rel_path,
                start_line=entry.start_line,
                end_line=entry.end_line,
                context="",
                metadata=_entry_metadata(entry),
            )
        )
    return documents


def _detail_pairs(metadata: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Text-mode lines for a PO hit: msgid, then msgstr with "(untranslated)"/"[fuzzy]"."""
    tags = "  [fuzzy]" if metadata.get("fuzzy") else ""
    msgstr = record_loaders.one_line(metadata.get("msgstr", "")) or "(untranslated)"
    return [
        ("msgid", record_loaders.one_line(metadata.get("msgid", ""))),
        ("msgstr", f"{msgstr}{tags}"),
    ]


record_loaders.register(
    record_loaders.RecordLoader(
        kind=PO_KIND,
        suffixes=PO_SUFFIXES,
        load=po_entries_to_documents,
        describe=_detail_pairs,
    )
)
