"""Tests for retrieval.record_loaders (registry, detail lines, chunk-loader routing)."""

import pathlib
import sys
import tempfile
import unittest
from typing import List

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval import project_loader, record_loaders  # noqa: E402
from retrieval.document import Document  # noqa: E402
from retrieval.record_loaders import RecordLoader  # noqa: E402

_PO_FIXTURE = _ROOT_DIR / "tests" / "fixtures" / "po" / "sample.po"
XYZ_KIND = "xyz_record"


def _load_xyz(docid: str, text: str, url: str) -> List[Document]:
    """Fake loader: one Document per non-blank line."""
    documents: List[Document] = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        documents.append(
            Document(
                docid=f"{docid}:{n}-{n}",
                text=line,
                url=url,
                source_path=docid,
                start_line=n,
                end_line=n,
                context="",
                metadata={"kind": XYZ_KIND, "title": line, "lang": "en"},
            )
        )
    return documents


def register_fake_xyz(case: unittest.TestCase) -> None:
    """Register the fake ``.xyz`` loader, unregistering it when *case* finishes."""
    record_loaders.register(
        RecordLoader(
            kind=XYZ_KIND,
            suffixes=frozenset({".xyz"}),
            load=_load_xyz,
            detail_keys=("title", "lang"),
        )
    )
    case.addCleanup(record_loaders.unregister, XYZ_KIND)


class TestRegistry(unittest.TestCase):
    def setUp(self) -> None:
        register_fake_xyz(self)

    def test_lookup_by_suffix_case_insensitive(self) -> None:
        for path in ("a/b.xyz", "B.XYZ"):
            loader = record_loaders.lookup(path)
            self.assertIsNotNone(loader)
            self.assertEqual(loader.kind, XYZ_KIND)
        self.assertIsNone(record_loaders.lookup("x.txt"))
        self.assertEqual(record_loaders.lookup("x.po").kind, "po_entry")

    def test_unregister_drops_only_that_kind(self) -> None:
        self.assertGreaterEqual(record_loaders.registered_suffixes(), {".po", ".xyz"})
        record_loaders.unregister(XYZ_KIND)
        suffixes = record_loaders.registered_suffixes()
        self.assertNotIn(".xyz", suffixes)
        self.assertIn(".po", suffixes)
        record_loaders.unregister("never-registered")  # no-op

    def test_conflicting_suffix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            record_loaders.register(
                RecordLoader(kind="other", suffixes=frozenset({".po"}), load=_load_xyz)
            )
        self.assertEqual(record_loaders.lookup("x.po").kind, "po_entry")

    def test_suffix_without_dot_rejected(self) -> None:
        with self.assertRaises(ValueError):
            record_loaders.register(
                RecordLoader(kind="nodot", suffixes=frozenset({"abc"}), load=_load_xyz)
            )

    def test_reregister_replaces_suffixes(self) -> None:
        record_loaders.register(
            RecordLoader(kind=XYZ_KIND, suffixes=frozenset({".abc"}), load=_load_xyz)
        )
        self.assertIsNone(record_loaders.lookup("a.xyz"))
        self.assertEqual(record_loaders.lookup("a.abc").kind, XYZ_KIND)

    def test_default_extensions_from_registry(self) -> None:
        self.assertIn(".po", project_loader.DEFAULT_EXTENSIONS)
        self.assertNotIn(".pot", project_loader.DEFAULT_EXTENSIONS)


class TestDetailLines(unittest.TestCase):
    def setUp(self) -> None:
        register_fake_xyz(self)

    def test_generic_follows_detail_keys(self) -> None:
        metadata = {"kind": XYZ_KIND, "lang": "en", "title": "a\nb", "extra": "x"}
        self.assertEqual(
            record_loaders.detail_lines(metadata), ["    title: a\\nb", "    lang: en"]
        )

    def test_missing_keys_skipped(self) -> None:
        self.assertEqual(
            record_loaders.detail_lines({"kind": XYZ_KIND, "lang": "en"}), ["    lang: en"]
        )

    def test_no_details_for_unknown_or_empty(self) -> None:
        for metadata in ({}, {"kind": "unknown"}, {"kind": ["x"]}):
            self.assertEqual(record_loaders.detail_lines(metadata), [])

    def test_po_translated(self) -> None:
        metadata = {"kind": "po_entry", "msgid": "Iterator Types", "msgstr": "疊代器型別"}
        self.assertEqual(
            record_loaders.detail_lines(metadata),
            ["    msgid: Iterator Types", "    msgstr: 疊代器型別"],
        )

    def test_po_fuzzy_and_untranslated(self) -> None:
        fuzzy = {"kind": "po_entry", "msgid": "a", "msgstr": "b", "fuzzy": True}
        self.assertEqual(record_loaders.detail_lines(fuzzy)[1], "    msgstr: b  [fuzzy]")
        untranslated = {"kind": "po_entry", "msgid": "a", "msgstr": ""}
        self.assertEqual(record_loaders.detail_lines(untranslated)[1], "    msgstr: (untranslated)")

    def test_one_line_truncates(self) -> None:
        self.assertEqual(record_loaders.one_line("x" * 200), "x" * 160 + "...")
        self.assertEqual(record_loaders.one_line("a\nb"), "a\\nb")


class TestLoaderRouting(unittest.TestCase):
    def setUp(self) -> None:
        register_fake_xyz(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        po_text = _PO_FIXTURE.read_text(encoding="utf-8")
        (self.root / "sample.po").write_text(po_text, encoding="utf-8")
        (self.root / "notes.md").write_text("# Notes\n\nSome prose here.\n", encoding="utf-8")
        (self.root / "data.xyz").write_text("first\n\nsecond\n", encoding="utf-8")

    def _check(self, documents: List[Document]) -> None:
        xyz = [d for d in documents if d.metadata.get("kind") == XYZ_KIND]
        self.assertEqual(len(xyz), 2)
        self.assertTrue(all(d.source_path == "data.xyz" for d in xyz))
        self.assertEqual(sum(d.metadata.get("kind") == "po_entry" for d in documents), 6)
        notes = [d for d in documents if d.source_path == "notes.md"]
        self.assertTrue(notes)
        self.assertTrue(all(d.metadata == {} for d in notes))

    def test_load_chunk_documents(self) -> None:
        extensions = project_loader.DEFAULT_EXTENSIONS | {".xyz"}
        self._check(project_loader.load_chunk_documents(self.root, extensions=extensions))

    def test_load_ast_chunk_documents(self) -> None:
        extensions = project_loader.DEFAULT_EXTENSIONS | {".xyz"}
        self._check(project_loader.load_ast_chunk_documents(self.root, extensions=extensions))


if __name__ == "__main__":
    unittest.main()
