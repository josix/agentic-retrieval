"""Tests for retrieval.project_loader.

Builds a fake project tree in a tmp directory with included files, excluded
directories, secret-like filenames, a binary file, and an oversize text file,
then verifies discover_files/read_text_safe/load_documents/load_chunks only
surface the intended, safe files with relative-path docids.
"""

import pathlib
import sys
import tempfile
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval import extractors  # noqa: E402
from retrieval.chunker import ChunkingPolicy  # noqa: E402
from retrieval.persistence import compute_fingerprint  # noqa: E402
from retrieval.project_loader import (  # noqa: E402
    DEFAULT_EXCLUDE_GLOBS,
    DEFAULT_EXTENSIONS,
    MAX_FILE_BYTES,
    discover_files,
    load_ast_chunk_documents,
    load_chunk_documents,
    load_chunks,
    load_documents,
    read_text_safe,
)

#: Guard: every secret-exclusion glob that existed before the hyperparameter-
#: plumbing work must still be present (constraint: never narrow the
#: secret-exclusion list). Add new patterns freely; never remove one below.
_ORIGINAL_EXCLUDE_GLOBS = (
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

try:
    import tree_sitter_language_pack  # noqa: E402,F401

    _TREESITTER_INSTALLED = True
except ImportError:
    _TREESITTER_INSTALLED = False


def _write(path: pathlib.Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class TestProjectLoader(unittest.TestCase):
    """Fake project fixture covering inclusion, exclusion, and safety rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.tmpdir.name)

        # --- included files ---
        _write(cls.root / "README.md", "# Project\n\nSome docs about routing.")
        _write(cls.root / "src" / "app.py", "def main():\n    pass\n")
        _write(cls.root / "notes.txt", "Some notes about the project.")

        # --- extensionless well-known files (basename allowlist) ---
        _write(cls.root / "Dockerfile", "FROM python:3.12-slim\n")
        _write(cls.root / "Makefile", "test:\n\tpytest\n")

        # --- excluded directories ---
        _write(cls.root / ".git" / "config", "[core]\nrepositoryformatversion = 0")
        _write(cls.root / "node_modules" / "x" / "index.js", "module.exports = {};")
        _write(cls.root / ".venv" / "lib" / "y.py", "# vendored\n")
        _write_bytes(cls.root / "__pycache__" / "z.pyc", b"\x00\x01\x02cachedbytes")
        _write(cls.root / ".complexipy_cache" / "README.md", "# cache artifact\n")
        _write(cls.root / "site" / "index.html", "<html>generated site output</html>")

        # --- secret-like filenames (allowed extension/name pattern, denied by glob) ---
        _write(cls.root / ".env", "SECRET_KEY=abc123")
        _write(cls.root / "server.pem", "-----BEGIN CERTIFICATE-----")
        _write(cls.root / "id_rsa", "-----BEGIN OPENSSH PRIVATE KEY-----")

        # --- binary file with an allowed extension ---
        _write_bytes(cls.root / "binary.py", b"\x00\x01\x02\x03binarydata")

        # --- oversize text file ---
        _write(cls.root / "huge.txt", "a" * (MAX_FILE_BYTES + 1))

        cls.discovered = discover_files(cls.root)
        cls.relative = {p.relative_to(cls.root).as_posix() for p in cls.discovered}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_includes_expected_files(self) -> None:
        expected = {
            "README.md",
            "src/app.py",
            "notes.txt",
            "binary.py",
            "Dockerfile",
            "Makefile",
        }
        self.assertEqual(self.relative, expected)

    def test_excludes_vcs_and_dependency_dirs(self) -> None:
        for bad in (
            ".git/config",
            "node_modules/x/index.js",
            ".venv/lib/y.py",
            "__pycache__/z.pyc",
            ".complexipy_cache/README.md",
            "site/index.html",
        ):
            self.assertNotIn(bad, self.relative)

    def test_excludes_secret_filenames(self) -> None:
        for bad in (".env", "server.pem", "id_rsa"):
            self.assertNotIn(bad, self.relative)

    def test_excludes_oversize_file(self) -> None:
        self.assertNotIn("huge.txt", self.relative)

    def test_docids_are_relative_posix_paths(self) -> None:
        for path in self.discovered:
            rel = path.relative_to(self.root).as_posix()
            self.assertFalse(rel.startswith("/"))

    def test_default_exclude_globs_contains_every_original_pattern(self) -> None:
        for pattern in _ORIGINAL_EXCLUDE_GLOBS:
            self.assertIn(pattern, DEFAULT_EXCLUDE_GLOBS)

    def test_empty_dir_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            self.assertEqual(discover_files(empty_dir), [])

    def test_read_text_safe_returns_none_for_binary(self) -> None:
        self.assertIsNone(read_text_safe(self.root / "binary.py"))

    def test_read_text_safe_returns_text_for_normal_file(self) -> None:
        text = read_text_safe(self.root / "notes.txt")
        self.assertEqual(text, "Some notes about the project.")

    def test_load_documents_docids_are_relative_paths(self) -> None:
        docs = load_documents(self.root)
        docids = {doc.docid for doc in docs}
        self.assertEqual(
            docids, {"README.md", "src/app.py", "notes.txt", "Dockerfile", "Makefile"}
        )
        for doc in docs:
            self.assertEqual(doc.url, "")

    def test_load_documents_skips_binary(self) -> None:
        docs = load_documents(self.root)
        docids = {doc.docid for doc in docs}
        self.assertNotIn("binary.py", docids)

    def test_load_chunks_doc_ids_are_relative_paths(self) -> None:
        chunks = load_chunks(self.root)
        self.assertGreater(len(chunks), 0)
        doc_ids = {chunk.doc_id for chunk in chunks}
        self.assertEqual(
            doc_ids, {"README.md", "src/app.py", "notes.txt", "Dockerfile", "Makefile"}
        )

    def test_oversize_basename_allowlisted_file_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "Dockerfile", "a" * (MAX_FILE_BYTES + 1))
            discovered = discover_files(root)
            self.assertEqual(discovered, [])

    def test_load_chunk_documents_docids_are_path_start_end(self) -> None:
        docs = load_chunk_documents(self.root)
        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertIn(":", doc.docid)
            path_part, span_part = doc.docid.rsplit(":", 1)
            start_str, end_str = span_part.split("-")
            self.assertEqual(path_part, doc.source_path)
            self.assertEqual(int(start_str), doc.start_line)
            self.assertEqual(int(end_str), doc.end_line)
            self.assertLessEqual(doc.start_line, doc.end_line)

    def test_load_chunk_documents_source_paths_match_load_documents(self) -> None:
        docs = load_chunk_documents(self.root)
        source_paths = {doc.source_path for doc in docs}
        whole_docids = {doc.docid for doc in load_documents(self.root)}
        self.assertEqual(source_paths, whole_docids)

    def test_load_chunk_documents_blank_file_yields_no_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "blank.txt", "   \n\n  \n")
            docs = load_chunk_documents(root)
            self.assertEqual(docs, [])

    def test_no_policy_output_identical_to_default_policy(self) -> None:
        no_policy_docids = {d.docid for d in load_chunk_documents(self.root)}
        default_policy_docids = {
            d.docid for d in load_chunk_documents(self.root, policy=ChunkingPolicy())
        }
        self.assertEqual(no_policy_docids, default_policy_docids)

    def test_larger_code_chars_yields_strictly_fewer_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(
                root / "big.py",
                "\n\n".join(f"def fn_{i}():\n    return {i}" for i in range(80)),
            )
            small = load_chunk_documents(root, policy=ChunkingPolicy(code_chars=40))
            large = load_chunk_documents(root, policy=ChunkingPolicy(code_chars=4000))
            self.assertGreater(len(small), len(large))

    def test_policy_and_discovery_kwargs_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "a.py", "x = 1\n")
            _write(root / "b.txt", "hello world\n")
            docs = load_chunk_documents(
                root, policy=ChunkingPolicy(code_chars=999), extensions=frozenset({".py"})
            )
            source_paths = {d.source_path for d in docs}
            self.assertEqual(source_paths, {"a.py"})

    def test_load_chunk_documents_carries_heading_as_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "doc.md", "## Heading\n\nSome body text under the heading.\n")
            docs = load_chunk_documents(root, extensions=frozenset({".md"}))
            self.assertGreater(len(docs), 0)
            self.assertTrue(any(d.context == "Heading" for d in docs))

    def test_load_chunk_documents_headingless_file_has_empty_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "plain.txt", "just some plain body text, no heading at all.\n")
            docs = load_chunk_documents(root, extensions=frozenset({".txt"}))
            self.assertGreater(len(docs), 0)
            for doc in docs:
                self.assertEqual(doc.context, "")


@unittest.skipUnless(_TREESITTER_INSTALLED, "tree-sitter-language-pack not installed")
class TestLoadAstChunkDocuments(unittest.TestCase):
    """load_ast_chunk_documents: AST chunks for code, line-chunk fallback
    for non-code files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.tmpdir.name)
        _write(
            cls.root / "src" / "app.py",
            "def foo():\n    return 1\n\n\nclass Bar:\n    def baz(self):\n        return 2\n",
        )
        _write(cls.root / "README.md", "# Project\n\nSome docs about routing.")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_docids_keep_path_start_end_convention(self) -> None:
        docs = load_ast_chunk_documents(self.root)
        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertIn(":", doc.docid)
            path_part, span_part = doc.docid.rsplit(":", 1)
            start_str, end_str = span_part.split("-")
            self.assertEqual(path_part, doc.source_path)
            self.assertEqual(int(start_str), doc.start_line)
            self.assertEqual(int(end_str), doc.end_line)

    def test_py_file_yields_chunks_some_with_context(self) -> None:
        docs = load_ast_chunk_documents(self.root)
        py_docs = [d for d in docs if d.source_path == "src/app.py"]
        self.assertGreater(len(py_docs), 0)
        # Small max_chars default (1200) keeps this snippet as one whole
        # chunk with an empty top-level context; force a smaller budget to
        # confirm nested chunks do carry a breadcrumb.
        from retrieval.ast_chunker import chunk_code

        small_chunks = chunk_code(
            "src/app.py",
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
            "python",
            max_chars=5,
        )
        contexts = {c.context for c in small_chunks}
        self.assertTrue(any(c for c in contexts if c))

    def test_md_file_falls_back_to_line_chunks_with_empty_context(self) -> None:
        docs = load_ast_chunk_documents(self.root)
        md_docs = [d for d in docs if d.source_path == "README.md"]
        self.assertGreater(len(md_docs), 0)
        for doc in md_docs:
            self.assertEqual(doc.context, "")


@unittest.skipIf(_TREESITTER_INSTALLED, "tree-sitter-language-pack is installed")
class TestLoadAstChunkDocumentsGracefulDegradation(unittest.TestCase):
    """A .py-only tree makes load_ast_chunk_documents raise, uncaught —
    that RuntimeError is the all-mode index skip signal (see cli._index_all)."""

    def test_raises_runtime_error_when_uninstalled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "app.py", "def foo():\n    return 1\n")
            with self.assertRaises(RuntimeError):
                load_ast_chunk_documents(root)


class TestPdfAutoActivation(unittest.TestCase):
    """WP-A: PDFs are discovered by default (no import-probing, no opt-in
    flag) — eligibility must stay deterministic regardless of whether the
    ``pdf`` extra happens to be installed in a given environment."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    def test_pdf_is_discovered_by_default(self) -> None:
        # T-P1 (inverted): a .pdf file IS discovered without any opt-in.
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "manual.pdf", pages=1)
            discovered = {p.relative_to(root).as_posix() for p in discover_files(root)}
            self.assertIn("manual.pdf", discovered)

    def test_discovery_and_fingerprint_identical_regardless_of_backend(self) -> None:
        # T-P1b: eligibility/fingerprinting never probes pypdf importability.
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "manual.pdf", pages=1)
            _write(root / "notes.txt", "some notes")
            files_before = sorted(p.relative_to(root).as_posix() for p in discover_files(root))
            fp_before = compute_fingerprint(root)

            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                files_after = sorted(
                    p.relative_to(root).as_posix() for p in discover_files(root)
                )
                fp_after = compute_fingerprint(root)
            finally:
                extractors.backend_available = original

            self.assertEqual(files_before, files_after)
            self.assertEqual(fp_before, fp_after)

    def test_extensions_override_minus_extractable_excludes_pdfs(self) -> None:
        # T-P2 (repurposed): an explicit `extensions=` narrower than
        # DEFAULT_EXTENSIONS still fully overrides (no implicit PDF re-add).
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "manual.pdf", pages=1)
            _write(root / "notes.txt", "some notes")
            narrow = DEFAULT_EXTENSIONS - extractors.EXTRACTABLE_EXTENSIONS
            discovered = {
                p.relative_to(root).as_posix() for p in discover_files(root, extensions=narrow)
            }
            self.assertNotIn("manual.pdf", discovered)
            self.assertIn("notes.txt", discovered)

    def test_two_stage_size_cap_pdf_vs_txt(self) -> None:
        # T-P3: a 2MB PDF is discovered (under extract_max_bytes), a 30MB
        # PDF is not (over it), and a 2MB .txt is not (over max_bytes).
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "small.pdf", pages=1)
            # pad small.pdf up to ~2MB by appending a big PDF comment.
            with open(root / "small.pdf", "ab") as fh:
                fh.write(b"\n%" + b"a" * (2_000_000 - (root / "small.pdf").stat().st_size))
            _write(root / "big.txt", "a" * 2_000_000)

            discovered = {p.relative_to(root).as_posix() for p in discover_files(root)}
            self.assertIn("small.pdf", discovered)
            self.assertNotIn("big.txt", discovered)

            discovered_tiny_cap = {
                p.relative_to(root).as_posix()
                for p in discover_files(root, extract_max_bytes=1_000)
            }
            self.assertNotIn("small.pdf", discovered_tiny_cap)

    def test_chunk_docids_and_spans_resolve_to_sidecar_lines(self) -> None:
        # T-P4: chunk docid uses the sidecar path; spans resolve to real
        # lines in the sidecar file on disk.
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "manual.pdf", pages=1)
            docs = load_chunk_documents(root, extensions=frozenset({".pdf"}))
            self.assertGreater(len(docs), 0)
            for doc in docs:
                self.assertTrue(
                    doc.docid.startswith(".agentic-retrieval/extracted/manual.pdf.md:")
                )
                self.assertEqual(doc.source_path, ".agentic-retrieval/extracted/manual.pdf.md")
                sidecar_path = root / doc.source_path
                self.assertTrue(sidecar_path.exists())
                lines = sidecar_path.read_text(encoding="utf-8").splitlines()
                self.assertLessEqual(doc.end_line, len(lines))
                self.assertGreaterEqual(doc.start_line, 1)

    @unittest.skipIf(_TREESITTER_INSTALLED, "tree-sitter-language-pack is installed")
    def test_ast_loader_falls_back_to_line_chunking_for_sidecar_docs(self) -> None:
        # T-P6: a sidecar's docid has no tree-sitter-mapped suffix (it's
        # ".md"), so load_ast_chunk_documents line-chunks it like any other
        # non-code file — this must not require the treesitter extra.
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "manual.pdf", pages=1)
            docs = load_ast_chunk_documents(root, extensions=frozenset({".pdf"}))
            self.assertGreater(len(docs), 0)
            for doc in docs:
                self.assertEqual(doc.context, "")

    def test_sidecar_is_never_itself_discovered(self) -> None:
        # T-P7: no double-indexing — the sidecar lives under the already-
        # excluded .agentic-retrieval dir.
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_pdf(root / "manual.pdf", pages=1)
            load_documents(root, extensions=frozenset({".pdf"}))  # writes the sidecar
            discovered = {p.relative_to(root).as_posix() for p in discover_files(root)}
            self.assertFalse(any(".agentic-retrieval" in d for d in discovered))

    def test_no_pdf_tree_discovery_and_fingerprint_unchanged(self) -> None:
        # T-P8: a tree with no PDFs discovers/fingerprints identically to
        # the pre-PDF extension set.
        pre_pdf_extensions = DEFAULT_EXTENSIONS - extractors.EXTRACTABLE_EXTENSIONS
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write(root / "a.py", "x = 1\n")
            _write(root / "b.md", "# doc\n\nsome text\n")
            with_pdf_default = sorted(
                p.relative_to(root).as_posix() for p in discover_files(root)
            )
            without_pdf_ext = sorted(
                p.relative_to(root).as_posix()
                for p in discover_files(root, extensions=pre_pdf_extensions)
            )
            self.assertEqual(with_pdf_default, without_pdf_ext)
            self.assertEqual(
                compute_fingerprint(root), compute_fingerprint(root, extensions=pre_pdf_extensions)
            )


class TestAgentOnlyMediaDiscovery(unittest.TestCase):
    """A ``.png``/``.docx`` under root is discovered like a PDF and routed
    through ``ensure_sidecar`` — the NUL-byte sniff in ``read_text_safe``
    must never be reached for these suffixes (it would classify them as
    binary and silently drop them)."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    def test_png_is_discovered_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "diagram.png").write_bytes(b"\x00not a real png\x00payload")
            discovered = {p.relative_to(root).as_posix() for p in discover_files(root)}
            self.assertIn("diagram.png", discovered)

    def test_docx_is_routed_through_ensure_sidecar_not_dropped_by_nul_sniff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "report.docx").write_bytes(b"\x00not a real docx\x00payload")
            documents = load_documents(root)
            docids = {doc.docid for doc in documents}
            self.assertIn(".agentic-retrieval/extracted/report.docx.md", docids)
            docx_sidecar = ".agentic-retrieval/extracted/report.docx.md"
            doc = next(d for d in documents if d.docid == docx_sidecar)
            self.assertIn("sidecar --register", doc.text)


class TestMediaTierDiscovery(unittest.TestCase):
    """Coverage for the three-tier media model's interaction with discovery:
    pairwise-disjoint suffix sets, union consistency, and the uncapped
    tier-3 size exemption."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    def test_tiers_are_pairwise_disjoint(self) -> None:
        self.assertEqual(
            extractors.MACHINE_EXTRACTABLE_EXTENSIONS & extractors.AGENT_ONLY_EXTENSIONS,
            frozenset(),
        )
        self.assertEqual(
            extractors.MACHINE_EXTRACTABLE_EXTENSIONS
            & extractors.AGENT_ORCHESTRATED_EXTENSIONS,
            frozenset(),
        )
        self.assertEqual(
            extractors.AGENT_ONLY_EXTENSIONS & extractors.AGENT_ORCHESTRATED_EXTENSIONS,
            frozenset(),
        )

    def test_extractable_extensions_is_the_three_tier_union(self) -> None:
        self.assertEqual(
            extractors.EXTRACTABLE_EXTENSIONS,
            extractors.MACHINE_EXTRACTABLE_EXTENSIONS
            | extractors.AGENT_ONLY_EXTENSIONS
            | extractors.AGENT_ORCHESTRATED_EXTENSIONS,
        )

    def test_large_mp4_discovered_while_similarly_large_pdf_is_not(self) -> None:
        from tests.test_extractors import _write_pdf

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mp4_path = root / "recording.mp4"
            mp4_path.write_bytes(b"\x00" * 40_000_000)  # 40MB, well over EXTRACT_MAX_BYTES

            pdf_path = root / "paper.pdf"
            _write_pdf(pdf_path, pages=1)
            with open(pdf_path, "ab") as fh:
                fh.write(b"\n%" + b"a" * (30_000_000 - pdf_path.stat().st_size))

            discovered = {p.relative_to(root).as_posix() for p in discover_files(root)}
            self.assertIn("recording.mp4", discovered)
            self.assertNotIn("paper.pdf", discovered)


if __name__ == "__main__":
    unittest.main()
