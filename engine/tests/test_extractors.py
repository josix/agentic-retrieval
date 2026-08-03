"""Tests for retrieval.extractors (PDF sidecar-transcript extraction).

Builds minimal ASCII PDF fixtures byte-literally (``_write_pdf``, no binary
files committed to the repo) and verifies path derivation, the content-hash
+ extractor-version manifest cache, the pypdf reflow pipeline, the
never-empty stub taxonomy, and the missing-backend guidance RuntimeError —
skipUnless/skipIf in both directions, mirroring
``tests/test_retrievers.py``'s ``_TURBOVEC_INSTALLED`` pattern.
"""

import ast
import inspect
import os
import pathlib
import sys
import tempfile
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from retrieval import extractors, persistence  # noqa: E402
from retrieval.chunker import chunk_document  # noqa: E402
from retrieval.project_loader import DEFAULT_EXCLUDE_DIRS  # noqa: E402

try:
    import pypdf  # noqa: F401

    _PYPDF_INSTALLED = True
except ImportError:
    _PYPDF_INSTALLED = False

#: A realistic multi-line paragraph, long enough per line that a 2-3 page
#: PDF built from it clears the >100-char/page "no-text-layer" threshold.
_DEFAULT_LINES = [
    "This is the first long line of a paragraph that keeps going on and on well past.",
    "This continues the same paragraph across a wrapped line boundary here wonder-",
    "ful indeed, spanning multiple lines to test reflow logic thoroughly across pages.",
]


def _write_pdf(path: pathlib.Path, pages: int = 1, lines=None, blank: bool = False) -> None:
    """Write a minimal, valid ASCII PDF (byte-literal, ~500B/page) to *path*.

    *lines* (default ``_DEFAULT_LINES``) are drawn on every page via
    successive ``Td``/``Tj`` operators (each line its own text-showing op,
    so pypdf's ``extract_text()`` joins them with ``\\n`` — this is what
    lets the de-hyphenation reflow test exercise a real wrapped word).
    *blank=True* emits an empty ``BT ET`` content stream (no text at all),
    for the no-text-layer stub case.
    """
    if lines is None:
        lines = _DEFAULT_LINES
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids_refs = " ".join(f"{3 + 2 * i} 0 R" for i in range(pages))
    objects.append(f"<< /Type /Pages /Kids [{kids_refs}] /Count {pages} >>".encode("ascii"))
    font_obj_num = 3 + 2 * pages
    for i in range(pages):
        content_obj_num = 4 + 2 * i
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                f"/Contents {content_obj_num} 0 R >>"
            ).encode("ascii")
        )
        if blank:
            stream_body = b"BT ET"
        else:
            parts = ["BT", "/F1 10 Tf", "20 260 Td"]
            for j, ln in enumerate(lines):
                if j > 0:
                    parts.append("0 -12 Td")
                parts.append(f"({ln}) Tj")
            parts.append("ET")
            stream_body = "\n".join(parts).encode("ascii")
        objects.append(
            b"<< /Length " + str(len(stream_body)).encode("ascii") + b" >>\nstream\n"
            + stream_body + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    body = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(body)
    n = len(objects) + 1
    body += f"xref\n0 {n}\n".encode("ascii") + b"0000000000 65535 f \n"
    for off in offsets:
        body += f"{off:010d} 00000 n \n".encode("ascii")
    body += (
        f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))


def _write_encrypted_pdf(path: pathlib.Path) -> None:
    """A one-page PDF encrypted with a non-empty user password — decrypting
    with the empty-password attempt ``_extract_pdf`` makes must fail."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="secret123", owner_password="ownerpw")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        writer.write(fh)


class TestPdfFixtureParses(unittest.TestCase):
    """Sanity-check the byte-literal fixture itself parses under pypdf."""

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_fixture_parses_and_extracts_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "doc.pdf"
            _write_pdf(path, pages=2)
            reader = pypdf.PdfReader(str(path))
            self.assertEqual(len(reader.pages), 2)
            self.assertIn("wonderful", reader.pages[0].extract_text().replace("-\n", ""))


class TestPathDerivation(unittest.TestCase):
    def test_sidecar_relpath_nested_dirs(self) -> None:
        self.assertEqual(
            extractors.sidecar_relpath("docs/sub/paper.pdf"),
            ".agentic-retrieval/extracted/docs/sub/paper.pdf.md",
        )

    def test_sidecar_relpath_top_level(self) -> None:
        self.assertEqual(
            extractors.sidecar_relpath("paper.pdf"),
            ".agentic-retrieval/extracted/paper.pdf.md",
        )

    def test_extract_dir_is_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.assertEqual(
                extractors.extract_dir(root), root.resolve() / ".agentic-retrieval" / "extracted"
            )

    def test_extract_dir_ignores_retrieval_index_dir_env(self) -> None:
        # T-E2: citation-contract regression — sidecars must stay under the
        # project root even when the *cache* is redirected elsewhere.
        old_env = os.environ.get("RETRIEVAL_INDEX_DIR")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as override:
            os.environ["RETRIEVAL_INDEX_DIR"] = override
            try:
                root = pathlib.Path(tmp)
                d = extractors.extract_dir(root)
            finally:
                if old_env is None:
                    os.environ.pop("RETRIEVAL_INDEX_DIR", None)
                else:
                    os.environ["RETRIEVAL_INDEX_DIR"] = old_env
            self.assertEqual(d, root.resolve() / ".agentic-retrieval" / "extracted")
            self.assertFalse(str(d).startswith(override))


class TestCacheDirnameDriftGuard(unittest.TestCase):
    def test_matches_persistence_cache_dirname_and_is_excluded_from_discovery(self) -> None:
        self.assertEqual(extractors._CACHE_DIRNAME, persistence.CACHE_DIRNAME)
        self.assertIn(extractors._CACHE_DIRNAME, DEFAULT_EXCLUDE_DIRS)


class TestModuleScopeIsStdlibOnly(unittest.TestCase):
    """T: no third-party imports at module scope, even when pypdf is
    installed in this environment — only inside function bodies."""

    _STDLIB_TOP_LEVEL = {
        "dataclasses", "hashlib", "io", "json", "os", "re", "statistics",
        "struct", "sys", "time", "unicodedata", "pathlib", "typing",
    }

    def test_no_third_party_imports_at_module_scope(self) -> None:
        source = inspect.getsource(extractors)
        tree = ast.parse(source)
        top_level_names = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_names.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_names.append(node.module.split(".")[0])
        self.assertNotIn("pypdf", top_level_names)
        for name in top_level_names:
            self.assertIn(name, self._STDLIB_TOP_LEVEL)


@unittest.skipIf(_PYPDF_INSTALLED, "pypdf is installed; guidance path not exercised")
class TestMissingPypdfGuidance(unittest.TestCase):
    def test_require_extractors_raises_guidance_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            extractors.require_extractors({".pdf"})
        self.assertIn(".[pdf]", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ImportError)

    def test_require_extractors_no_op_for_non_pdf_extensions(self) -> None:
        extractors.require_extractors({".txt", ".md"})  # must not raise

    def test_backend_available_false(self) -> None:
        self.assertFalse(extractors.backend_available())

    def test_pdf_processed_without_backend_yields_backend_missing_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            extractors.clear_process_cache()
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.status, "stub")
            self.assertEqual(sidecar.reason, "backend-missing")
            chunks = chunk_document(sidecar.docid, sidecar.text)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(any("doc.pdf" in c.text for c in chunks))


class TestManifestCorruption(unittest.TestCase):
    def test_corrupt_manifest_treated_as_empty_no_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            directory = extractors.extract_dir(root)
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("{not valid json", encoding="utf-8")
            self.assertEqual(extractors.load_manifest(root), {})

    def test_missing_manifest_needs_reextraction_false_without_stat_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.assertFalse(extractors.needs_reextraction(root))


@unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
class TestExtractionPipeline(unittest.TestCase):
    """Requires a real pypdf backend to parse the byte-literal fixtures."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    def test_ok_extraction_yields_page_headings_and_dehyphenated_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=2)
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.status, "ok")
            self.assertTrue(sidecar.text.startswith("<!-- source: doc.pdf -->"))
            self.assertIn("## Page 1", sidecar.text)
            self.assertIn("## Page 2", sidecar.text)
            self.assertIn("wonderful", sidecar.text)  # de-hyphenated
            self.assertNotIn("wonder-\nful", sidecar.text)

    def test_ok_sidecar_yields_multiple_chunks_at_prose_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=3)
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            chunks = chunk_document(sidecar.docid, sidecar.text, target_chars=400)
            self.assertGreater(len(chunks), 1)

    def test_docid_matches_sidecar_relpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "sub" / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.docid, ".agentic-retrieval/extracted/sub/doc.pdf.md")
            self.assertTrue(sidecar.path.exists())

    def test_determinism_two_calls_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=2)
            s1 = extractors.ensure_sidecar(root, pdf_path, force=True)
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, pdf_path, force=True)
            self.assertEqual(s1.text, s2.text)

    def test_cache_hit_on_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            s1 = extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(s1.cache_hit)
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, pdf_path)
            self.assertTrue(s2.cache_hit)
            self.assertEqual(s2.text, s1.text)

    def test_cache_hit_survives_mtime_bump_with_unchanged_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            extractors.ensure_sidecar(root, pdf_path)
            new_ns = pdf_path.stat().st_mtime_ns + 1_000_000_000
            os.utime(pdf_path, ns=(new_ns, new_ns))
            extractors.clear_process_cache()
            hit = extractors.ensure_sidecar(root, pdf_path)
            self.assertTrue(hit.cache_hit)

    def test_miss_on_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            s1 = extractors.ensure_sidecar(root, pdf_path)
            _write_pdf(pdf_path, pages=2)
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(s2.cache_hit)
            self.assertNotEqual(s1.text, s2.text)

    def test_miss_on_extractor_version_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            extractors.ensure_sidecar(root, pdf_path)
            manifest = extractors.load_manifest(root)
            manifest["entries"]["doc.pdf"]["extractor_version"] = "stale/0"
            extractors.save_manifest(root, manifest)
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(s2.cache_hit)

    def test_deleted_sidecar_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            s1 = extractors.ensure_sidecar(root, pdf_path)
            s1.path.unlink()
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(s2.cache_hit)
            self.assertTrue(s2.path.exists())

    def test_truncated_sidecar_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            s1 = extractors.ensure_sidecar(root, pdf_path)
            s1.path.write_text("truncated garbage", encoding="utf-8")
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(s2.cache_hit)
            self.assertEqual(s2.text, s1.text)

    def test_encrypted_pdf_yields_stub_naming_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "locked.pdf"
            _write_encrypted_pdf(pdf_path)
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.status, "stub")
            self.assertEqual(sidecar.reason, "encrypted")
            chunks = chunk_document(sidecar.docid, sidecar.text)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(any("locked.pdf" in c.text for c in chunks))

    def test_malformed_pdf_yields_stub_naming_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "bad.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nthis is not a real pdf body at all")
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.status, "stub")
            self.assertEqual(sidecar.reason, "malformed")
            chunks = chunk_document(sidecar.docid, sidecar.text)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(any("bad.pdf" in c.text for c in chunks))

    def test_empty_pdf_yields_stub_naming_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "empty.pdf"
            _write_pdf(pdf_path, pages=0)
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.status, "stub")
            self.assertEqual(sidecar.reason, "empty")
            chunks = chunk_document(sidecar.docid, sidecar.text)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(any("empty.pdf" in c.text for c in chunks))

    def test_no_text_layer_pdf_yields_stub_naming_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "scanned.pdf"
            _write_pdf(pdf_path, pages=1, blank=True)
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertEqual(sidecar.status, "stub")
            self.assertEqual(sidecar.reason, "no-text-layer")
            chunks = chunk_document(sidecar.docid, sidecar.text)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(any("scanned.pdf" in c.text for c in chunks))

    def test_truncation_at_page_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "big.pdf"
            _write_pdf(pdf_path, pages=3)
            old_budget = extractors.MAX_TRANSCRIPT_CHARS
            extractors.MAX_TRANSCRIPT_CHARS = 400
            try:
                sidecar = extractors.ensure_sidecar(root, pdf_path, force=True)
            finally:
                extractors.MAX_TRANSCRIPT_CHARS = old_budget
            self.assertEqual(sidecar.status, "ok")
            self.assertIn("truncated", sidecar.text)
            manifest = extractors.load_manifest(root)
            self.assertTrue(manifest["entries"]["big.pdf"]["truncated"])
            # cut at a whole page boundary, never mid-page
            self.assertTrue(sidecar.text.count("## Page") < 3)

    def test_backend_missing_entry_is_miss_once_backend_becomes_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)

            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                stub = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertEqual(stub.reason, "backend-missing")

            extractors.clear_process_cache()
            regenerated = extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(regenerated.cache_hit)
            self.assertEqual(regenerated.status, "ok")

    def test_ok_entry_stays_hit_when_backend_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            extractors.ensure_sidecar(root, pdf_path)

            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                still_hit = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertTrue(still_hit.cache_hit)
            self.assertEqual(still_hit.status, "ok")

    def test_needs_reextraction_true_for_backend_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertTrue(extractors.needs_reextraction(root))

    def test_needs_reextraction_false_after_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            extractors.clear_process_cache()
            extractors.ensure_sidecar(root, pdf_path)
            self.assertFalse(extractors.needs_reextraction(root))


class TestReflowPipeline(unittest.TestCase):
    """White-box tests for the private reflow helpers."""

    def test_normalize_text_dehyphenates_wrapped_words(self) -> None:
        # De-hyphenation runs on the raw multi-line string (_normalize_text),
        # upstream of the already-line-split _reflow_page.
        normalized = extractors._normalize_text("wonder-\nful reflow across pages")
        self.assertIn("wonderful", normalized)
        self.assertNotIn("wonder-\nful", normalized)

    def test_multiple_paragraphs_from_blank_line_separated_blocks(self) -> None:
        lines = [
            "wonderful reflow across a wrapped word boundary right here for real",
            "",
            "short",
            "A brand new second paragraph after the blank-line separator above here.",
        ]
        body = extractors._reflow_page(lines)
        paragraphs = body.split("\n\n")
        self.assertGreater(len(paragraphs), 1)

    def test_short_line_ends_paragraph_mid_block(self) -> None:
        lines = [
            "This is a long first line that sets a high median length for the page.",
            "This is another long line similar in length to the one directly above.",
            "short line",
            "This restarts as a new paragraph after the short line ended the last one.",
        ]
        body = extractors._reflow_page(lines)
        paragraphs = [p for p in body.split("\n\n") if p]
        self.assertGreater(len(paragraphs), 1)

    def test_subthreecharacter_paragraphs_are_dropped(self) -> None:
        lines = ["1", "", "A real paragraph with enough content to survive the length filter."]
        body = extractors._reflow_page(lines)
        self.assertNotIn("1\n\n", body)

    def test_whitespace_collapse(self) -> None:
        lines = ["A   line     with   many      spaces   between   words   here for length."]
        body = extractors._reflow_page(lines)
        self.assertNotIn("   ", body)

    def test_blank_input_returns_empty_string(self) -> None:
        self.assertEqual(extractors._reflow_page(["", "  ", ""]), "")


class TestHeaderFooterStripping(unittest.TestCase):
    def test_repeated_first_line_stripped_when_doc_has_5plus_pages(self) -> None:
        pages_lines = [["RUNNING HEADER", f"body text for page {i}"] for i in range(6)]
        stripped = extractors._strip_headers_footers(pages_lines)
        for lines in stripped:
            self.assertNotIn("RUNNING HEADER", lines)

    def test_no_stripping_under_5_pages(self) -> None:
        pages_lines = [["RUNNING HEADER", f"body text for page {i}"] for i in range(3)]
        stripped = extractors._strip_headers_footers(pages_lines)
        self.assertEqual(stripped, pages_lines)


class TestTruncateAtPageBoundary(unittest.TestCase):
    def test_cuts_at_whole_page_boundary_under_budget(self) -> None:
        page_bodies = [f"## Page {i}\n\n" + ("word " * 50) for i in range(1, 6)]
        old_budget = extractors.MAX_TRANSCRIPT_CHARS
        extractors.MAX_TRANSCRIPT_CHARS = 400
        try:
            transcript, truncated = extractors._truncate_at_page_boundary(page_bodies)
        finally:
            extractors.MAX_TRANSCRIPT_CHARS = old_budget
        self.assertTrue(truncated)
        self.assertIn("truncated", transcript)
        self.assertTrue(transcript.count("## Page") < len(page_bodies))

    def test_always_keeps_at_least_one_page(self) -> None:
        page_bodies = ["## Page 1\n\n" + ("word " * 5000)]
        old_budget = extractors.MAX_TRANSCRIPT_CHARS
        extractors.MAX_TRANSCRIPT_CHARS = 10
        try:
            transcript, truncated = extractors._truncate_at_page_boundary(page_bodies)
        finally:
            extractors.MAX_TRANSCRIPT_CHARS = old_budget
        self.assertTrue(truncated)
        self.assertIn("## Page 1", transcript)


if __name__ == "__main__":
    unittest.main()
