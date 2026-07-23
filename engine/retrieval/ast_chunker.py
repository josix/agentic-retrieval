"""AST-boundary ("cAST") chunking for code/script corpora via tree-sitter.

Implements the split-then-merge chunking strategy from cAST (arXiv
2506.15655): recursively walk a file's parse tree, greedily merging
consecutive sibling nodes into a chunk while the combined non-whitespace
character count stays within ``max_chars``; a node that alone exceeds the
budget is recursed into instead of merged; a leaf node (no named children)
that still exceeds the budget is hard-split by lines. Each emitted chunk
carries a 1-based ``[start_line, end_line]`` span (same convention as
``retrieval.chunker.chunk_document``) plus a dotted breadcrumb of enclosing
scope names (``context``, e.g. ``"Bar.baz"``) collected only from ancestor
nodes the chunker actually recursed into (so a whole small file, or a
function kept intact as one chunk, gets an empty ``context``).

Lazy-imports ``tree_sitter_language_pack``; the package is optional (the
``treesitter`` extra) so importing this module never requires it — only
calling ``chunk_code`` does, mirroring the guidance-``RuntimeError``
convention in ``retrieval.retrievers`` (see ``_require_backends``).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

#: File suffix -> tree-sitter-language-pack language identifier. Best-effort,
#: Python-first; unmapped suffixes (including non-code text like .md/.txt)
#: return None from ``language_for_path`` so the loader falls back to the
#: line-based ``chunk_document``.
LANGUAGE_BY_SUFFIX: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
}

#: Language -> set of node types that introduce a named scope for the
#: breadcrumb (best-effort per language; grammars vary in node naming).
SCOPE_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"class_declaration", "interface_declaration", "method_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item", "impl_item", "struct_item"},
    "ruby": {"method", "class", "module"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "c": {"function_definition"},
    "cpp": {"function_definition", "class_specifier"},
    "csharp": {"class_declaration", "method_declaration"},
    "kotlin": {"function_declaration", "class_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition"},
    "swift": {"function_declaration", "class_declaration"},
    "lua": {"function_declaration"},
    "bash": {"function_definition"},
}

_WS_RE = re.compile(r"\s+")


@dataclass
class AstChunk:
    text: str
    start_line: int
    end_line: int
    context: str


def language_for_path(path: str) -> Optional[str]:
    """Return the tree-sitter language identifier for *path*'s suffix, or
    ``None`` if unmapped (the caller should fall back to line chunking)."""
    for suffix, language in LANGUAGE_BY_SUFFIX.items():
        if path.lower().endswith(suffix):
            return language
    return None


def _nonws_len(text: str) -> int:
    return len(_WS_RE.sub("", text))


def _node_text(node, raw_bytes: bytes) -> str:
    return raw_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _scope_name(node, scope_types: Set[str]) -> Optional[str]:
    """Return the breadcrumb name for *node* if its type is a named scope."""
    if node.type not in scope_types:
        return None
    name_node = node.child_by_field_name("name")
    return name_node.text.decode("utf-8", errors="replace") if name_node else node.type


def _is_severely_broken(root) -> bool:
    """True when the parse failed so badly the tree is unusable for chunking
    (the root itself is an ERROR node, or every named top-level child is)."""
    if root.type == "ERROR":
        return True
    named_children = [c for c in root.children if c.is_named]
    if not named_children:
        return False
    return all(c.type == "ERROR" for c in named_children)


def _emit_group_chunk(nodes: List, raw_bytes: bytes, ancestors: List[str],
                       chunks: List[AstChunk]) -> None:
    if not nodes:
        return
    text = raw_bytes[nodes[0].start_byte : nodes[-1].end_byte].decode(
        "utf-8", errors="replace"
    )
    chunks.append(
        AstChunk(
            text=text,
            start_line=nodes[0].start_point.row + 1,
            end_line=nodes[-1].end_point.row + 1,
            context=".".join(ancestors),
        )
    )


def _hard_split_leaf(node, raw_bytes: bytes, max_chars: int,
                      ancestors: List[str]) -> List[AstChunk]:
    """Line-split a leaf node (no named children) that still exceeds max_chars."""
    text = _node_text(node, raw_bytes)
    lines = text.splitlines()
    context = ".".join(ancestors)
    chunks: List[AstChunk] = []
    buf: List[str] = []
    buf_chars = 0
    buf_start = node.start_point.row + 1
    line_no = buf_start
    for line in lines:
        line_chars = _nonws_len(line)
        if buf and buf_chars + line_chars > max_chars:
            chunks.append(
                AstChunk(
                    text="\n".join(buf), start_line=buf_start,
                    end_line=line_no - 1, context=context,
                )
            )
            buf = []
            buf_chars = 0
            buf_start = line_no
        buf.append(line)
        buf_chars += line_chars
        line_no += 1
    if buf:
        chunks.append(
            AstChunk(
                text="\n".join(buf), start_line=buf_start,
                end_line=line_no - 1, context=context,
            )
        )
    return chunks


def _partition_siblings(siblings: List, raw_bytes: bytes,
                         max_chars: int) -> List[Tuple[str, object]]:
    """Greedily partition *siblings* into ``("group", [nodes])`` (fits within
    *max_chars* together) / ``("oversized", node)`` (alone exceeds it) items,
    preserving order."""
    groups: List[Tuple[str, object]] = []
    current: List = []
    current_chars = 0
    for node in siblings:
        node_chars = _nonws_len(_node_text(node, raw_bytes))
        if node_chars > max_chars:
            if current:
                groups.append(("group", current))
                current, current_chars = [], 0
            groups.append(("oversized", node))
            continue
        if current and current_chars + node_chars > max_chars:
            groups.append(("group", current))
            current, current_chars = [], 0
        current.append(node)
        current_chars += node_chars
    if current:
        groups.append(("group", current))
    return groups


def _merge_small_trailing_group(groups: List[Tuple[str, object]], raw_bytes: bytes,
                                 max_chars: int) -> List[Tuple[str, object]]:
    """Fold a too-small trailing ``"group"`` item into the one before it,
    instead of emitting a standalone tiny chunk."""
    if len(groups) < 2 or groups[-1][0] != "group" or groups[-2][0] != "group":
        return groups
    last_chars = sum(_nonws_len(_node_text(n, raw_bytes)) for n in groups[-1][1])
    if last_chars >= max_chars * 0.25:
        return groups
    merged = groups[:-2] + [("group", groups[-2][1] + groups[-1][1])]
    return merged


def _group_siblings(siblings: List, raw_bytes: bytes,
                     max_chars: int) -> List[Tuple[str, object]]:
    """Partition *siblings* into ``("group", [nodes])`` / ``("oversized",
    node)`` items, in order, then merge a too-small trailing group into the
    previous one instead of emitting a standalone tiny chunk."""
    groups = _partition_siblings(siblings, raw_bytes, max_chars)
    return _merge_small_trailing_group(groups, raw_bytes, max_chars)


def _process_siblings(siblings: List, raw_bytes: bytes, max_chars: int,
                       scope_types: Set[str], ancestors: List[str],
                       chunks: List[AstChunk]) -> None:
    for kind, payload in _group_siblings(siblings, raw_bytes, max_chars):
        if kind == "group":
            _emit_group_chunk(payload, raw_bytes, ancestors, chunks)
            continue
        node = payload
        name = _scope_name(node, scope_types)
        new_ancestors = ancestors + [name] if name else ancestors
        children = [c for c in node.children if c.is_named]
        if children:
            _process_siblings(children, raw_bytes, max_chars, scope_types, new_ancestors, chunks)
        else:
            chunks.extend(_hard_split_leaf(node, raw_bytes, max_chars, new_ancestors))


def chunk_code(doc_id: str, raw_text: str, language: str,
                max_chars: int = 1200) -> List[AstChunk]:
    """Split *raw_text* (source for *doc_id*, parsed as *language*) into
    ``AstChunk``s at AST node boundaries.

    Raises ``RuntimeError`` with opt-in guidance if the ``treesitter`` extra
    isn't installed. Returns ``[]`` (signaling "fall back to line chunking")
    when *language*'s grammar isn't available in the installed language pack,
    when the text is blank, or when the parse is too broken to chunk
    meaningfully.
    """
    del doc_id  # kept in the signature to match chunk_document's shape
    if not raw_text.strip():
        return []

    try:
        # NOTE: pyproject.toml's `treesitter` extra pins `tree-sitter<0.26`.
        # tree-sitter==0.26.0 has a native memory-corruption regression that
        # segfaults (SIGSEGV) deterministically after a handful of sequential
        # parser.parse() calls in one process — it is *not* fixed by
        # restructuring this function to avoid retaining Node objects (that
        # was tried and still crashed); it is an upstream bug isolated to
        # tree-sitter 0.26.0 specifically (0.24.0-0.25.2 are clean, and the
        # tree-sitter-language-pack version is not implicated). Do not raise
        # the upper bound without re-verifying against the sequential-parse
        # regression test in tests/test_ast_chunker.py.
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:  # pragma: no cover - guidance path
        raise RuntimeError(
            "tree-sitter retriever needs the 'treesitter' extra:\n"
            "  uv pip install -e '.[treesitter]'"
        ) from exc

    try:
        parser = get_parser(language)
    except Exception:
        # Grammar unavailable in the installed pack; caller falls back to
        # the line chunker for this file.
        return []

    raw_bytes = raw_text.encode("utf-8")
    tree = parser.parse(raw_bytes)
    root = tree.root_node
    if _is_severely_broken(root):
        return []

    scope_types = SCOPE_NODE_TYPES.get(language, set())
    siblings = [c for c in root.children if c.is_named]
    if not siblings:
        return []

    chunks: List[AstChunk] = []
    _process_siblings(siblings, raw_bytes, max_chars, scope_types, [], chunks)
    return chunks
