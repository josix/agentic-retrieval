"""Tests for retrieval.extractors (PDF sidecar-transcript extraction).

Builds minimal ASCII PDF fixtures byte-literally (``_write_pdf``, no binary
files committed to the repo) and verifies path derivation, the content-hash
+ extractor-version manifest cache, the pypdf reflow pipeline, the
never-empty stub taxonomy, and the missing-backend guidance RuntimeError —
skipUnless/skipIf in both directions, mirroring
``tests/test_retrievers.py``'s ``_TURBOVEC_INSTALLED`` pattern.
"""

import ast
import hashlib
import inspect
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

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
        self.assertIn("sidecar --register", str(ctx.exception))
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

    def test_backend_missing_stub_mentions_sidecar_register(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            extractors.clear_process_cache()
            sidecar = extractors.ensure_sidecar(root, pdf_path)
            self.assertIn("sidecar --register", sidecar.text)


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

    def test_entry_missing_status_key_is_cache_miss_not_keyerror(self) -> None:
        # A manifest entry missing "status" (hand-edited/corrupted, or from
        # a future schema) must fall through _is_cache_hit's membership
        # check to a cache miss + re-extraction, not raise KeyError from
        # ensure_sidecar's cache-hit branch (which used to index
        # entry["status"] unconditionally).
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            sidecar_rel = extractors.sidecar_relpath("doc.pdf")
            sidecar_path = root / sidecar_rel
            sidecar_path.parent.mkdir(parents=True)
            sidecar_path.write_text("stub content", encoding="utf-8")
            manifest_dir = extractors.extract_dir(root)
            (manifest_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "entries": {
                            "doc.pdf": {
                                "sha256": sha256,
                                "extractor_version": extractors.EXTRACTOR_VERSION,
                                "sidecar": sidecar_rel,
                                "sidecar_bytes": len("stub content".encode("utf-8")),
                                # "status" intentionally omitted.
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            extractors.clear_process_cache()
            sidecar = extractors.ensure_sidecar(root, pdf_path)  # must not raise
            self.assertIsNotNone(sidecar.status)


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
            transcript, truncated = extractors._truncate_at_unit_boundary(page_bodies)
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
            transcript, truncated = extractors._truncate_at_unit_boundary(page_bodies)
        finally:
            extractors.MAX_TRANSCRIPT_CHARS = old_budget
        self.assertTrue(truncated)
        self.assertIn("## Page 1", transcript)


class TestAgentAuthoredSidecar(unittest.TestCase):
    """Coverage for the no-pypdf agent-authored sidecar path
    (``register_sidecar``) — must pass with or without pypdf installed,
    since ``register_sidecar`` itself never imports the backend."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    def test_register_sidecar_writes_entry_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            sidecar = extractors.register_sidecar(root, pdf_path, "## Page 1\n\nHello world.")
            self.assertEqual(sidecar.status, "ok")
            self.assertTrue(sidecar.path.exists())
            entry = extractors.load_manifest(root)["entries"]["doc.pdf"]
            self.assertEqual(entry["extractor_version"], extractors.AGENT_EXTRACTOR_VERSION)
            self.assertEqual(entry["authored_by"], "agent")
            self.assertEqual(entry["status"], "ok")
            self.assertEqual(entry["reason"], "")
            self.assertIn("authored_at", entry)
            self.assertIn("sidecar_sha256", entry)
            self.assertEqual(entry["pages"], 1)

    def test_register_sidecar_header_names_agent_extractor_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            sidecar = extractors.register_sidecar(root, pdf_path, "some transcript text")
            self.assertIn(f"extractor: {extractors.AGENT_EXTRACTOR_VERSION}", sidecar.text)

    def test_register_sidecar_preserves_page_headings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            transcript = "## Page 1\n\nfirst.\n\n## Page 2\n\nsecond."
            sidecar = extractors.register_sidecar(root, pdf_path, transcript)
            self.assertIn("## Page 1", sidecar.text)
            self.assertIn("## Page 2", sidecar.text)
            entry = extractors.load_manifest(root)["entries"]["doc.pdf"]
            self.assertEqual(entry["pages"], 2)

    def test_registered_sidecar_is_cache_hit_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            extractors.register_sidecar(root, pdf_path, "hand-authored transcript")
            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                sidecar = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertTrue(sidecar.cache_hit)
            self.assertEqual(sidecar.status, "ok")

    def test_registered_sidecar_is_cache_hit_with_backend_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            extractors.register_sidecar(root, pdf_path, "hand-authored transcript")
            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: True
            try:
                sidecar = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertTrue(sidecar.cache_hit)
            self.assertEqual(sidecar.status, "ok")

    def test_ensure_sidecar_does_not_overwrite_agent_sidecar_with_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            extractors.register_sidecar(root, pdf_path, "hand-authored transcript")
            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                sidecar = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertNotEqual(sidecar.reason, "backend-missing")
            self.assertIn("hand-authored transcript", sidecar.text)

    def test_register_sidecar_rejects_empty_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            with self.assertRaises(ValueError):
                extractors.register_sidecar(root, pdf_path, "   \n  ")

    def test_register_sidecar_rejects_non_extractable_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            txt_path = root / "doc.txt"
            txt_path.write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError):
                extractors.register_sidecar(root, txt_path, "transcript")

    def test_register_sidecar_rejects_source_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            root = pathlib.Path(tmp1)
            outside = pathlib.Path(tmp2) / "doc.pdf"
            outside.write_bytes(b"%PDF-1.4\nnot a real body")
            with self.assertRaises(ValueError):
                extractors.register_sidecar(root, outside, "transcript")

    def test_register_sidecar_strips_leading_header_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            transcript = "<!-- source: doc.pdf -->\n<!-- extractor: x -->\nActual content."
            sidecar = extractors.register_sidecar(root, pdf_path, transcript)
            self.assertEqual(sidecar.text.count("<!-- source:"), 1)
            self.assertIn("Actual content.", sidecar.text)

    def test_register_sidecar_truncates_oversized_transcript_at_page_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            transcript = "\n\n".join(
                f"## Page {i}\n\n" + ("word " * 50) for i in range(1, 6)
            )
            old_budget = extractors.MAX_TRANSCRIPT_CHARS
            extractors.MAX_TRANSCRIPT_CHARS = 400
            try:
                sidecar = extractors.register_sidecar(root, pdf_path, transcript)
            finally:
                extractors.MAX_TRANSCRIPT_CHARS = old_budget
            self.assertIn("truncated", sidecar.text)
            entry = extractors.load_manifest(root)["entries"]["doc.pdf"]
            self.assertTrue(entry["truncated"])
            self.assertTrue(sidecar.text.count("## Page") < 5)

    def test_register_sidecar_invalidates_process_memo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                stub = extractors.ensure_sidecar(root, pdf_path)
                self.assertEqual(stub.reason, "backend-missing")
                extractors.register_sidecar(root, pdf_path, "hand-authored transcript")
                # No clear_process_cache() call here: register_sidecar must
                # invalidate the memo itself, else this call would keep
                # serving the stale stub keyed by the source's unchanged
                # (size, mtime) rather than the freshly registered sidecar.
                sidecar = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertEqual(sidecar.status, "ok")
            self.assertIn("hand-authored transcript", sidecar.text)

    def test_source_change_invalidates_agent_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nfirst body")
            extractors.register_sidecar(root, pdf_path, "hand-authored transcript")
            pdf_path.write_bytes(b"%PDF-1.4\nsecond, different body content here")
            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                sidecar = extractors.ensure_sidecar(root, pdf_path)
            finally:
                extractors.backend_available = original
            self.assertFalse(sidecar.cache_hit)
            self.assertEqual(sidecar.reason, "backend-missing")

    def test_needs_reextraction_false_for_agent_authored_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nnot a real body")
            extractors.register_sidecar(root, pdf_path, "hand-authored transcript")
            original = extractors.backend_available
            extractors.backend_available = lambda: True
            try:
                self.assertFalse(extractors.needs_reextraction(root))
            finally:
                extractors.backend_available = original

    def test_agent_sidecar_revision_empty_for_pypdf_entry(self) -> None:
        pypdf_entry = {"authored_by": "not-an-agent", "sidecar_sha256": "deadbeef"}
        self.assertEqual(extractors.agent_sidecar_revision(pypdf_entry), "")
        self.assertEqual(extractors.agent_sidecar_revision({}), "")


class TestAgentOnlyMedia(unittest.TestCase):
    """Media suffixes with no machine extractor at all (docx/pptx/xlsx and
    images) — ``ensure_sidecar`` always writes an ``"agent-only"`` stub for
    these, regardless of ``backend_available()``, and that stub must never
    be mistaken for a ``backend-missing`` one (which self-heals once pypdf
    is installed; an agent-only stub never should)."""

    def setUp(self) -> None:
        extractors.clear_process_cache()
        extractors._WARNED_AGENT_ONLY = False

    def tearDown(self) -> None:
        extractors.clear_process_cache()
        extractors._WARNED_AGENT_ONLY = False

    def test_extension_sets_are_consistent(self) -> None:
        expected_agent_only = {
            ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".gif", ".webp",
        }
        self.assertEqual(extractors.AGENT_ONLY_EXTENSIONS, frozenset(expected_agent_only))
        self.assertEqual(
            extractors.MACHINE_EXTRACTABLE_EXTENSIONS, frozenset({".pdf", ".srt", ".vtt"})
        )
        self.assertEqual(
            extractors.EXTRACTABLE_EXTENSIONS,
            extractors.MACHINE_EXTRACTABLE_EXTENSIONS
            | extractors.AGENT_ONLY_EXTENSIONS
            | extractors.AGENT_ORCHESTRATED_EXTENSIONS,
        )
        self.assertTrue(expected_agent_only.issubset(extractors.EXTRACTABLE_EXTENSIONS))

    def _stub_for(self, suffix: str, *, backend_flag: bool) -> "extractors.Sidecar":
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            media_path = root / f"file{suffix}"
            media_path.write_bytes(b"not a real payload, just bytes")
            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: backend_flag
            try:
                return extractors.ensure_sidecar(root, media_path)
            finally:
                extractors.backend_available = original

    def test_docx_stub_reason_and_message_backend_unavailable(self) -> None:
        sidecar = self._stub_for(".docx", backend_flag=False)
        self.assertEqual(sidecar.status, "stub")
        self.assertEqual(sidecar.reason, "agent-only")
        self.assertIn("sidecar --register", sidecar.text)

    def test_png_stub_reason_and_message_backend_available(self) -> None:
        # Even with a pypdf backend installed/available, agent-only suffixes
        # never get routed through a pypdf-oriented extractor.
        sidecar = self._stub_for(".png", backend_flag=True)
        self.assertEqual(sidecar.status, "stub")
        self.assertEqual(sidecar.reason, "agent-only")
        self.assertIn("sidecar --register", sidecar.text)

    def test_agent_only_reason_is_never_backend_missing(self) -> None:
        sidecar = self._stub_for(".docx", backend_flag=False)
        self.assertNotEqual(sidecar.reason, "backend-missing")

    def test_one_time_stderr_note_fires_once_per_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            a = root / "a.png"
            b = root / "b.png"
            a.write_bytes(b"aaa")
            b.write_bytes(b"bbb")
            extractors.clear_process_cache()
            buf = io.StringIO()
            with redirect_stderr(buf):
                extractors.ensure_sidecar(root, a)
                extractors.ensure_sidecar(root, b)
            err = buf.getvalue()
            self.assertEqual(err.count("agent-transcribable stubs"), 1)

    def test_needs_reextraction_stays_false_for_agent_only_stub_with_backend_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            media_path = root / "diagram.png"
            media_path.write_bytes(b"not a real payload")
            extractors.clear_process_cache()
            original = extractors.backend_available
            extractors.backend_available = lambda: False
            try:
                extractors.ensure_sidecar(root, media_path)
            finally:
                extractors.backend_available = original
            # Now flip the backend "on" and confirm the agent-only stub is
            # never treated as a stale backend-missing PDF stub that a
            # newly available pypdf should self-heal.
            original = extractors.backend_available
            extractors.backend_available = lambda: True
            try:
                self.assertFalse(extractors.needs_reextraction(root))
            finally:
                extractors.backend_available = original

    def test_register_sidecar_accepts_docx_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            docx_path = root / "report.docx"
            docx_path.write_bytes(b"not a real docx payload")
            sidecar = extractors.register_sidecar(root, docx_path, "Hand-authored docx text.")
            self.assertEqual(sidecar.status, "ok")
            self.assertIn("Hand-authored docx text.", sidecar.text)

    def test_register_sidecar_accepts_png_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            png_path = root / "diagram.png"
            png_path.write_bytes(b"not a real png payload")
            sidecar = extractors.register_sidecar(
                root, png_path, "A diagram showing three connected boxes."
            )
            self.assertEqual(sidecar.status, "ok")
            self.assertIn("three connected boxes", sidecar.text)

    def test_require_extractors_no_op_for_agent_only_only_extension_set(self) -> None:
        # None of these need a backend: there isn't one for any of them.
        extractors.require_extractors(extractors.AGENT_ONLY_EXTENSIONS)  # must not raise


def _write_srt(path: pathlib.Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body.encode("utf-8"))


class TestCaptionExtraction(unittest.TestCase):
    """Coverage for ``_extract_captions`` (.srt/.vtt tier-1 parsing) — must
    pass with or without pypdf installed, since captions never touch that
    backend at all (mirrors ``TestAgentAuthoredSidecar``'s independence)."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    _SRT = (
        "1\n00:00:01,000 --> 00:00:04,000\nHello there, this is the first cue.\n\n"
        "2\n00:00:04,200 --> 00:00:07,000\nAnd this is a second, closely-following cue.\n"
    )
    _VTT = (
        "WEBVTT\n\n"
        "1\n00:00:01.000 --> 00:00:04.000\nHello there, this is the first cue.\n\n"
        "2\n00:00:04.200 --> 00:00:07.000\nAnd this is a second, closely-following cue.\n"
    )

    def test_srt_and_vtt_produce_byte_identical_bodies(self) -> None:
        srt = extractors._extract_captions(self._SRT.encode("utf-8"))
        vtt = extractors._extract_captions(self._VTT.encode("utf-8"))
        self.assertEqual(srt.status, "ok")
        self.assertEqual(srt.text, vtt.text)

    def test_strips_markup_and_ass_override_tags(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:04,000\n{\\an8}<b>Bold</b> and <i>italic</i> text.\n"
        r = extractors._extract_captions(srt.encode("utf-8"))
        self.assertIn("Bold and italic text.", r.text)
        self.assertNotIn("<b>", r.text)
        self.assertNotIn("{\\an8}", r.text)

    def test_voice_tag_becomes_speaker_prefix(self) -> None:
        vtt = (
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\n<v Alice>Hello everyone here today.</v>\n"
        )
        r = extractors._extract_captions(vtt.encode("utf-8"))
        self.assertIn("Alice: Hello everyone here today.", r.text)

    def test_comma_and_dot_decimal_timestamps_both_parse(self) -> None:
        srt_comma = "1\n00:00:01,500 --> 00:00:02,500\nComma decimal cue.\n"
        vtt_dot = "1\n00:00:01.500 --> 00:00:02.500\nDot decimal cue.\n"
        r1 = extractors._extract_captions(srt_comma.encode("utf-8"))
        r2 = extractors._extract_captions(vtt_dot.encode("utf-8"))
        self.assertEqual(r1.status, "ok")
        self.assertEqual(r2.status, "ok")
        self.assertIn("[00:00:01]", r1.text)
        self.assertIn("[00:00:01]", r2.text)

    def test_mm_ss_short_form_timestamp_parses(self) -> None:
        vtt = "WEBVTT\n\n1\n01:30.000 --> 01:35.000\nShort-form timestamp cue.\n"
        r = extractors._extract_captions(vtt.encode("utf-8"))
        self.assertEqual(r.status, "ok")
        self.assertIn("[00:01:30]", r.text)

    def test_vtt_cue_settings_are_ignored(self) -> None:
        vtt = (
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000 align:middle line:90%\n"
            "Cue with trailing settings.\n"
        )
        r = extractors._extract_captions(vtt.encode("utf-8"))
        self.assertEqual(r.status, "ok")
        self.assertIn("Cue with trailing settings.", r.text)

    def test_consecutive_duplicate_cues_are_collapsed(self) -> None:
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nRepeated line of text here today.\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nRepeated line of text here today.\n\n"
            "3\n00:00:03,000 --> 00:00:04,000\nA genuinely different final cue.\n"
        )
        cues, _saw_arrow = extractors._iter_caption_cues(srt)
        self.assertEqual(len(cues), 3)  # tokenizer sees all three raw cues...
        r = extractors._extract_captions(srt.encode("utf-8"))
        # ...but the extractor's dedup pass collapses the back-to-back
        # repeat before merging: the repeated sentence is never immediately
        # followed by itself again in the body.
        self.assertNotIn(
            "today. Repeated line of text here today.", r.text
        )
        self.assertIn("A genuinely different final cue.", r.text)

    def test_paragraphs_are_timestamp_prefixed(self) -> None:
        r = extractors._extract_captions(self._SRT.encode("utf-8"))
        self.assertIn("[00:00:01]", r.text)

    def test_new_section_after_180_seconds(self) -> None:
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nFirst section opening cue right here.\n\n"
            "2\n00:03:05,000 --> 00:03:06,000\nSecond section far enough later on.\n"
        )
        r = extractors._extract_captions(srt.encode("utf-8"))
        self.assertEqual(r.pages, 2)
        self.assertIn("## [00:00:01]", r.text)
        self.assertIn("## [00:03:05]", r.text)

    def test_zero_cues_yields_no_cues_stub(self) -> None:
        vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\n\n"
        r = extractors._extract_captions(vtt.encode("utf-8"))
        self.assertEqual(r.status, "stub")
        self.assertEqual(r.reason, "no-cues")

    def test_no_arrow_lines_yields_malformed_stub(self) -> None:
        r = extractors._extract_captions(b"this file has no cue timing lines at all")
        self.assertEqual(r.status, "stub")
        self.assertEqual(r.reason, "malformed")

    def test_cp1252_bytes_decode_without_raising(self) -> None:
        body = (
            "1\n00:00:01,000 --> 00:00:04,000\nCaf\xe9 with a non-UTF8 byte.\n"
        ).encode("cp1252")
        r = extractors._extract_captions(body)
        self.assertEqual(r.status, "ok")

    def test_pages_field_counts_sections(self) -> None:
        r = extractors._extract_captions(self._SRT.encode("utf-8"))
        self.assertEqual(r.pages, 1)

    def test_determinism_two_calls_identical_output(self) -> None:
        r1 = extractors._extract_captions(self._SRT.encode("utf-8"))
        r2 = extractors._extract_captions(self._SRT.encode("utf-8"))
        self.assertEqual(r1.text, r2.text)

    def test_captions_extractor_version_stamped_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            srt_path = root / "talk.srt"
            _write_srt(srt_path, self._SRT)
            extractors.ensure_sidecar(root, srt_path)
            entry = extractors.load_manifest(root)["entries"]["talk.srt"]
            self.assertEqual(entry["extractor_version"], extractors.CAPTIONS_EXTRACTOR_VERSION)

    def test_captions_extractor_version_stamped_in_sidecar_header(self) -> None:
        # Regression: _render_sidecar was called without version= at its
        # ensure_sidecar call site, so every sidecar's on-disk
        # "<!-- extractor: ... -->" header comment was stamped
        # pypdf-text/1 even for captions (the manifest's own
        # extractor_version was correct; only the rendered file's header
        # comment lied about its provenance).
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            srt_path = root / "talk.srt"
            _write_srt(srt_path, self._SRT)
            sidecar = extractors.ensure_sidecar(root, srt_path)
            self.assertIn(f"extractor: {extractors.CAPTIONS_EXTRACTOR_VERSION}", sidecar.text)
            self.assertNotIn(f"extractor: {extractors.EXTRACTOR_VERSION}", sidecar.text)

    def test_cache_hit_on_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            srt_path = root / "talk.srt"
            _write_srt(srt_path, self._SRT)
            s1 = extractors.ensure_sidecar(root, srt_path)
            self.assertFalse(s1.cache_hit)
            extractors.clear_process_cache()
            s2 = extractors.ensure_sidecar(root, srt_path)
            self.assertTrue(s2.cache_hit)

    def test_works_without_pypdf_installed(self) -> None:
        # Mirrors TestMissingPypdfGuidance's _no_pypdf idiom: captions never
        # touch pypdf, so this must pass regardless of what's installed.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            srt_path = root / "talk.srt"
            _write_srt(srt_path, self._SRT)
            extractors.clear_process_cache()
            sidecar = extractors.ensure_sidecar(root, srt_path)
            self.assertEqual(sidecar.status, "ok")


class TestUnitHeadingRegex(unittest.TestCase):
    def test_matches_page_heading(self) -> None:
        self.assertTrue(extractors._UNIT_HEADING_RE.search("## Page 7\n"))

    def test_matches_timestamp_heading(self) -> None:
        self.assertTrue(extractors._UNIT_HEADING_RE.search("## [01:02:03] Some topic\n"))

    def test_rejects_prose_subheading(self) -> None:
        self.assertFalse(extractors._UNIT_HEADING_RE.search("## Introduction\n"))

    def test_rejects_non_numeric_page(self) -> None:
        self.assertFalse(extractors._UNIT_HEADING_RE.search("## Page seven\n"))

    def test_rejects_h3_timestamp(self) -> None:
        self.assertFalse(extractors._UNIT_HEADING_RE.search("### [00:00:00] x\n"))

    def test_truncation_at_timestamp_boundary(self) -> None:
        segments = [f"## [00:0{i}:00] section\n\n" + ("word " * 50) for i in range(5)]
        old_budget = extractors.MAX_TRANSCRIPT_CHARS
        extractors.MAX_TRANSCRIPT_CHARS = 400
        try:
            transcript, truncated = extractors._truncate_at_unit_boundary(segments)
        finally:
            extractors.MAX_TRANSCRIPT_CHARS = old_budget
        self.assertTrue(truncated)
        self.assertTrue(transcript.count("## [00:0") < 5)


class TestStatOnlyIdentity(unittest.TestCase):
    """Coverage for tier-3 (``AGENT_ORCHESTRATED_EXTENSIONS``) stat-only
    source identity — the highest-risk change in this batch: the engine
    must never read a video/audio file's bytes."""

    def setUp(self) -> None:
        extractors.clear_process_cache()

    def test_ensure_sidecar_never_calls_read_bytes_on_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mp4_path = root / "talk.mp4"
            mp4_path.write_bytes(b"fake video payload, not really an mp4")
            extractors.clear_process_cache()

            original_read_bytes = pathlib.Path.read_bytes

            def _boom(self, *a, **k):
                raise AssertionError(f"read_bytes() called on {self}")

            pathlib.Path.read_bytes = _boom
            try:
                sidecar = extractors.ensure_sidecar(root, mp4_path)
            finally:
                pathlib.Path.read_bytes = original_read_bytes
            self.assertEqual(sidecar.status, "stub")
            self.assertEqual(sidecar.reason, "agent-orchestrated")

    def test_identity_field_is_stat_1_for_tier3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mp4_path = root / "talk.mp4"
            mp4_path.write_bytes(b"fake video payload")
            extractors.ensure_sidecar(root, mp4_path)
            entry = extractors.load_manifest(root)["entries"]["talk.mp4"]
            self.assertEqual(entry["identity"], extractors.IDENTITY_STAT)

    def test_register_then_ensure_is_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mp4_path = root / "talk.mp4"
            mp4_path.write_bytes(b"fake video payload")
            extractors.register_sidecar(
                root, mp4_path, "## [00:00:00] Intro\n\n[00:00:00] Hello from ASR."
            )
            extractors.clear_process_cache()
            sidecar = extractors.ensure_sidecar(root, mp4_path)
            self.assertTrue(sidecar.cache_hit)
            self.assertEqual(sidecar.status, "ok")

    def test_utime_bump_invalidates_stat_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mp4_path = root / "talk.mp4"
            mp4_path.write_bytes(b"fake video payload")
            extractors.ensure_sidecar(root, mp4_path)
            new_ns = mp4_path.stat().st_mtime_ns + 1_000_000_000
            os.utime(mp4_path, ns=(new_ns, new_ns))
            extractors.clear_process_cache()
            sidecar = extractors.ensure_sidecar(root, mp4_path)
            self.assertFalse(sidecar.cache_hit)

    def test_sidecar_list_does_not_read_bytes_for_tier3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mp4_path = root / "talk.mp4"
            mp4_path.write_bytes(b"fake video payload")
            extractors.ensure_sidecar(root, mp4_path)
            extractors.clear_process_cache()

            original_read_bytes = pathlib.Path.read_bytes

            def _boom(self, *a, **k):
                raise AssertionError(f"read_bytes() called on {self}")

            pathlib.Path.read_bytes = _boom
            try:
                states = extractors.sidecar_states(root, [mp4_path])
            finally:
                pathlib.Path.read_bytes = original_read_bytes
            self.assertEqual(states[0]["state"], "stub")

    @unittest.skipUnless(_PYPDF_INSTALLED, "pypdf not installed")
    def test_pdf_still_hash_stable_sha256_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pdf_path = root / "doc.pdf"
            _write_pdf(pdf_path, pages=1)
            extractors.ensure_sidecar(root, pdf_path)
            entry = extractors.load_manifest(root)["entries"]["doc.pdf"]
            self.assertEqual(entry["identity"], extractors.IDENTITY_SHA256)
            self.assertEqual(entry["sha256"], hashlib.sha256(pdf_path.read_bytes()).hexdigest())


class TestRequireExtractorsSplit(unittest.TestCase):
    """``require_extractors`` only cares about ``PYPDF_EXTENSIONS`` now —
    every other extractable tier (captions, agent-only, agent-orchestrated)
    is a no-op regardless of whether pypdf is installed."""

    def test_no_op_for_captions_only(self) -> None:
        extractors.require_extractors({".srt"})  # must not raise
        extractors.require_extractors({".vtt"})  # must not raise

    def test_no_op_for_agent_orchestrated_only(self) -> None:
        extractors.require_extractors(extractors.AGENT_ORCHESTRATED_EXTENSIONS)  # must not raise

    @unittest.skipIf(_PYPDF_INSTALLED, "pypdf installed; guidance path not exercised")
    def test_raises_for_pdf_extension(self) -> None:
        with self.assertRaises(RuntimeError):
            extractors.require_extractors({".pdf"})


if __name__ == "__main__":
    unittest.main()
