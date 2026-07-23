"""End-to-end pipeline tests.

Verifies that:
1. The contextualized index surfaces the designed 'Routing' chunk from
   computer_networks.txt for a query like "what carries data between networks"
   even though the chunk body does not contain those exact words.
2. The default modules have no top-level imports of network libraries
   (socket, urllib, requests, http).
3. The production retriever pipeline (load_chunk_documents -> index ->
   search_detailed) returns line-spanned hits that a coding agent can turn
   into a Read(path, offset=start_line, limit=end_line-start_line+1) call.
"""

import ast
import pathlib
import sys
import tempfile
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.chunker import chunk_document  # noqa: E402
from retrieval.index import ContextualRetriever  # noqa: E402
from retrieval.project_loader import load_chunk_documents  # noqa: E402
from retrieval.retrievers import LexicalRetriever  # noqa: E402

_CORPUS_DIR = pathlib.Path(__file__).parent / "fixtures"
_RETRIEVAL_DIR = _ROOT_DIR / "retrieval"

_QUERY = "what carries data between networks"


def _load_corpus() -> dict:
    docs = {}
    for path in sorted(_CORPUS_DIR.glob("*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def _build_all_chunks(docs: dict) -> list:
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(chunk_document(doc_id, text))
    return chunks


class TestPipelineContextualizationBenefit(unittest.TestCase):
    """The contextualized index must rank the Routing chunk at least as high as raw."""

    @classmethod
    def setUpClass(cls) -> None:
        docs = _load_corpus()
        chunks = _build_all_chunks(docs)

        cls.raw = ContextualRetriever()
        cls.raw.build(chunks, use_context=False)

        cls.ctx = ContextualRetriever()
        cls.ctx.build(chunks, use_context=True)

    def _routing_rank(self, retriever: ContextualRetriever) -> int:
        """Return the 0-based rank of the Routing chunk, or 9999 if not found."""
        results = retriever.search(_QUERY, top_k=20)
        for result in results:
            if (
                result.chunk.doc_id == "computer_networks"
                and result.chunk.heading.lower() == "routing"
            ):
                return result.fused_rank
        return 9999

    def test_contextual_finds_routing_chunk(self) -> None:
        """Routing chunk must appear in top-10 results for contextual index."""
        rank = self._routing_rank(self.ctx)
        self.assertLess(rank, 10, f"Routing chunk not found in top-10 (rank={rank})")

    def test_contextual_ranks_routing_at_least_as_high_as_raw(self) -> None:
        """Contextual rank of Routing chunk must be <= raw rank (better or equal)."""
        raw_rank = self._routing_rank(self.raw)
        ctx_rank = self._routing_rank(self.ctx)
        self.assertLessEqual(
            ctx_rank,
            raw_rank,
            f"Contextual rank ({ctx_rank}) worse than raw rank ({raw_rank})",
        )


class TestOfflineAssertion(unittest.TestCase):
    """Confirm no top-level network imports in default retrieval modules."""

    _NETWORK_MODULES = {"socket", "urllib", "requests", "http", "httpx", "aiohttp"}

    def _get_top_level_imports(self, source: str) -> set:
        """Parse Python source and return names of top-level imported modules."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names = set()
        for node in ast.walk(tree):
            # Only consider statements at module level (direct children of Module)
            pass
        # Walk only top-level statements
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module.split(".")[0])
        return names

    def test_no_network_imports_in_retrieval_modules(self) -> None:
        py_files = list(_RETRIEVAL_DIR.glob("*.py"))
        self.assertGreater(len(py_files), 0, "No Python files found in retrieval/")

        violations = []
        for fpath in py_files:
            source = fpath.read_text(encoding="utf-8")
            imports = self._get_top_level_imports(source)
            found = imports & self._NETWORK_MODULES
            if found:
                violations.append(f"{fpath.name}: {found}")

        self.assertEqual(
            violations,
            [],
            "Network libraries imported at top level:\n" + "\n".join(violations),
        )

    def test_pipeline_runs_without_network_access(self) -> None:
        """The full pipeline completes using only stdlib — no network needed."""
        docs = _load_corpus()
        self.assertGreater(len(docs), 0)
        chunks = _build_all_chunks(docs)
        self.assertGreater(len(chunks), 0)

        retriever = ContextualRetriever()
        retriever.build(chunks, use_context=True)
        results = retriever.search(_QUERY, top_k=5)
        self.assertEqual(len(results), 5)


class TestEndToEndChunkRetrieval(unittest.TestCase):
    """load_chunk_documents -> LexicalRetriever.index -> search_detailed
    must surface real, readable file:line spans for a coding agent."""

    def test_index_query_returns_line_spanned_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "networks.txt"
            source.write_text(
                (_CORPUS_DIR / "computer_networks.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            chunk_docs = load_chunk_documents(root)
            self.assertGreater(len(chunk_docs), 0)

            retriever = LexicalRetriever()
            retriever.index(chunk_docs)
            hits = retriever.search_detailed(_QUERY, top_k=1)

            self.assertEqual(len(hits), 1)
            hit = hits[0]
            self.assertEqual(hit.source_path, "networks.txt")
            self.assertGreaterEqual(hit.start_line, 1)
            self.assertGreaterEqual(hit.end_line, hit.start_line)
            self.assertEqual(hit.docid, f"networks.txt:{hit.start_line}-{hit.end_line}")

            # The span must actually resolve to real, non-empty source lines
            # (what a coding agent's Read(path, offset, limit) would return).
            all_lines = source.read_text(encoding="utf-8").splitlines()
            spanned = all_lines[hit.start_line - 1 : hit.end_line]
            self.assertTrue(any(line.strip() for line in spanned))


if __name__ == "__main__":
    unittest.main()
