"""Tests for retrieval.po_loader (per-entry PO parsing and Document building)."""

import pathlib
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval.po_loader import (  # noqa: E402
    is_po_path,
    parse_po,
    po_entries_to_documents,
    unescape_po_string,
)

FIXTURE = _ROOT_DIR / "tests" / "fixtures" / "po" / "sample.po"


class TestUnescape(unittest.TestCase):
    def test_common_escapes(self) -> None:
        self.assertEqual(unescape_po_string(r'"a\nb\tc\"d\\e"'), 'a\nb\tc"d\\e')

    def test_unknown_escape_is_kept(self) -> None:
        self.assertEqual(unescape_po_string(r'"a\qb"'), "a\\qb")

    def test_chinese_and_zero_width_separator_pass_through(self) -> None:
        self.assertEqual(unescape_po_string('"參見\\\\ :class:"'), "參見\\ :class:")

    def test_unterminated_quote_does_not_raise(self) -> None:
        self.assertEqual(unescape_po_string('"abc'), "abc")


class TestParsePo(unittest.TestCase):
    def setUp(self) -> None:
        self.text = FIXTURE.read_text(encoding="utf-8")
        self.entries = parse_po(self.text)

    def test_entry_count_and_spans(self) -> None:
        spans = [(e.start_line, e.end_line, e.msgid_line) for e in self.entries]
        self.assertEqual(
            spans,
            [(1, 6, 2), (8, 10, 9), (12, 18, 13), (20, 23, 22), (25, 27, 26),
             (29, 31, 30), (33, 35, 33)],
        )

    def test_header_detected(self) -> None:
        self.assertTrue(self.entries[0].is_header)
        self.assertFalse(any(e.is_header for e in self.entries[1:]))

    def test_continuations_joined(self) -> None:
        entry = self.entries[2]
        self.assertTrue(entry.msgid.startswith("Python supports"))
        self.assertTrue(entry.msgid.endswith("two distinct methods."))
        self.assertEqual(
            entry.msgstr, "Python 支援對容器進行疊代的概念。這是透過兩個不同的方法來實作的。"
        )

    def test_fuzzy_references_and_escapes(self) -> None:
        entry = self.entries[3]
        self.assertTrue(entry.fuzzy)
        self.assertEqual(entry.references, ["../../library/functions.rst:5"])
        self.assertIn('"frozenset"', entry.msgid)
        self.assertIn("\n", entry.msgid)
        self.assertIn("參見\\ :class:", entry.msgstr)

    def test_untranslated_msgctxt_and_plural(self) -> None:
        self.assertFalse(self.entries[4].translated)
        self.assertEqual(self.entries[5].msgctxt, "decorator")
        plural = self.entries[6]
        self.assertEqual(plural.msgid_plural, "%d files")
        self.assertEqual(plural.msgstr_plural, ["%d 個檔案"])
        self.assertTrue(plural.translated)

    def test_no_obsolete_text(self) -> None:
        for entry in self.entries:
            self.assertNotIn("Obsolete", entry.msgid)
            self.assertNotIn("已廢棄", entry.msgstr)

    def test_entries_without_blank_separators(self) -> None:
        text = 'msgid "a"\nmsgstr "A"\n#: x:1\nmsgid "b"\nmsgstr "B"\nmsgid "c"\nmsgstr "C"\n'
        entries = parse_po(text)
        self.assertEqual([e.msgid for e in entries], ["a", "b", "c"])
        self.assertEqual([(e.start_line, e.end_line) for e in entries], [(1, 2), (3, 5), (6, 7)])

    def test_crlf_and_bom_give_same_entries(self) -> None:
        plain = [(e.msgid, e.start_line, e.end_line) for e in self.entries]
        mangled = "﻿" + self.text.replace("\n", "\r\n")
        again = [(e.msgid, e.start_line, e.end_line) for e in parse_po(mangled)]
        self.assertEqual(plain, again)

    def test_malformed_input_does_not_raise(self) -> None:
        parse_po('msgid "unterminated\nstray text\nmsgstr\n"orphan"\n#~\n')
        parse_po("")


class TestPoEntriesToDocuments(unittest.TestCase):
    def setUp(self) -> None:
        text = FIXTURE.read_text(encoding="utf-8")
        self.docs = po_entries_to_documents("sample.po", text)

    def test_header_and_obsolete_skipped(self) -> None:
        self.assertEqual(len(self.docs), 6)

    def test_first_document_shape(self) -> None:
        doc = self.docs[0]
        self.assertEqual(doc.docid, "sample.po:8-10")
        self.assertEqual((doc.source_path, doc.start_line, doc.end_line), ("sample.po", 8, 10))
        self.assertIn("Iterator Types", doc.text)
        self.assertIn("疊代器型別", doc.text)
        self.assertEqual(doc.context, "")
        self.assertEqual(
            doc.metadata,
            {
                "kind": "po_entry",
                "msgid": "Iterator Types",
                "msgstr": "疊代器型別",
                "fuzzy": False,
                "translated": True,
                "line": 9,
                "references": ["../../library/stdtypes.rst:10"],
            },
        )

    def test_untranslated_entry(self) -> None:
        meta = self.docs[3].metadata
        self.assertFalse(meta["translated"])
        self.assertEqual(meta["msgstr"], "")

    def test_optional_metadata_keys(self) -> None:
        self.assertEqual(self.docs[4].metadata["msgctxt"], "decorator")
        self.assertNotIn("msgctxt", self.docs[0].metadata)
        self.assertEqual(self.docs[5].metadata["msgstr_plural"], ["%d 個檔案"])


class TestIsPoPath(unittest.TestCase):
    def test_suffixes(self) -> None:
        self.assertTrue(is_po_path("library/os.po"))
        self.assertTrue(is_po_path("X.PO"))
        self.assertFalse(is_po_path("library/os.pot"))
        self.assertFalse(is_po_path("a.txt"))


if __name__ == "__main__":
    unittest.main()
