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

from retrieval.project_loader import (  # noqa: E402
    MAX_FILE_BYTES,
    discover_files,
    load_chunk_documents,
    load_chunks,
    load_documents,
    read_text_safe,
)


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


if __name__ == "__main__":
    unittest.main()
