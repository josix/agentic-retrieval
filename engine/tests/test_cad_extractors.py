"""Tests for retrieval.cad_extractors and its wiring into retrieval.extractors.

IDS tests always run (stdlib XML). DXF/IFC tests need the ``cad``/``bim``
extras and skip without them; the missing-backend stub paths are exercised
in every environment by swapping the registered probe.
"""

import dataclasses
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from retrieval import cad_extractors, extractors  # noqa: E402
from retrieval.project_loader import discover_files, load_documents  # noqa: E402

try:
    import ezdxf

    HAVE_EZDXF = True
except ImportError:
    HAVE_EZDXF = False

try:
    import ifcopenshell
    import ifcopenshell.api

    HAVE_IFC = True
except ImportError:
    HAVE_IFC = False

IDS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ids xmlns="http://standards.buildingsmart.org/IDS" xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <info><title>Door requirements</title><author>qa@example.com</author></info>
  <specifications>
    <specification name="Door fire rating" ifcVersion="IFC4">
      <applicability minOccurs="1" maxOccurs="unbounded">
        <entity><name><simpleValue>IFCDOOR</simpleValue></name></entity>
      </applicability>
      <requirements>
        <property dataType="IFCLABEL" cardinality="required">
          <propertySet><simpleValue>Pset_DoorCommon</simpleValue></propertySet>
          <baseName><simpleValue>FireRating</simpleValue></baseName>
          <value><xs:restriction base="xs:string">
            <xs:enumeration value="30min"/><xs:enumeration value="60min"/>
          </xs:restriction></value>
        </property>
      </requirements>
    </specification>
  </specifications>
</ids>"""


def make_plate(path: pathlib.Path, units: int = 4, note: str = "4X DIA 8 THRU") -> None:
    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = units
    for name in ("PROFILE", "HOLES", "NOTE"):
        doc.layers.new(name)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True,
                       dxfattribs={"layer": "PROFILE"})
    for x, y in [(15, 15), (85, 15), (15, 45), (85, 45)]:
        msp.add_circle((x, y), 3, dxfattribs={"layer": "HOLES"})
    msp.add_text(note, dxfattribs={"height": 4, "insert": (4, 70), "layer": "NOTE"})
    block = doc.blocks.new("DOOR")
    block.add_line((0, 0), (900, 0))
    msp.add_blockref("DOOR", (10, 0))
    doc.saveas(path)


def make_ifc(path: pathlib.Path) -> None:
    run = ifcopenshell.api.run
    model = run("project.create_file", version="IFC4")
    run("root.create_entity", model, ifc_class="IfcProject", name="Training building")
    run("unit.assign_unit", model)
    storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="L1")
    for name, rating in [("D01", "60min"), ("D02", None)]:
        door = run("root.create_entity", model, ifc_class="IfcDoor", name=name)
        run("spatial.assign_container", model, products=[door], relating_structure=storey)
        if rating:
            pset = run("pset.add_pset", model, product=door, name="Pset_DoorCommon")
            run("pset.edit_pset", model, pset=pset, properties={"FireRating": rating})
    model.write(str(path))


class TestIds(unittest.TestCase):
    def test_describes_specification_in_words(self) -> None:
        result = cad_extractors.extract_ids(IDS_XML)
        self.assertEqual(result.status, "ok")
        for needle in ("Door requirements", "Door fire rating", "IFCDOOR", "Pset_DoorCommon",
                       "FireRating", "one of `30min`, `60min`", "required (at least one"):
            self.assertIn(needle, result.text)

    def test_malformed_xml_is_a_stub(self) -> None:
        result = cad_extractors.extract_ids(b"<ids><unclosed>")
        self.assertEqual((result.status, result.reason), ("stub", "malformed"))

    def test_ids_needs_no_backend(self) -> None:
        self.assertTrue(extractors.backend_ready(".ids"))


@unittest.skipUnless(HAVE_EZDXF, "needs the cad extra (ezdxf)")
class TestDxf(unittest.TestCase):
    def summarize(self, **kwargs) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "plate.dxf"
            make_plate(path, **kwargs)
            result = cad_extractors.extract_dxf(path.read_bytes())
        self.assertEqual(result.status, "ok")
        return result.text

    def test_reports_layers_text_blocks_and_real_area(self) -> None:
        text = self.summarize()
        self.assertIn("millimeters ($INSUNITS=4)", text)
        self.assertIn("| HOLES | 4 | CIRCLE 4 | visible |", text)
        self.assertIn("4X DIA 8 THRU", text)
        self.assertIn("| DOOR | 1 |", text)
        self.assertIn("| PROFILE | 6000 | 0.006 m2 |", text)

    def test_flags_hole_note_that_disagrees_with_geometry(self) -> None:
        text = self.summarize()
        self.assertIn("CONFLICT", text)
        self.assertIn("4 x diameter 6", text)

    def test_matching_hole_note_is_not_a_conflict(self) -> None:
        text = self.summarize(note="4X DIA 6 THRU")
        self.assertIn("matches the geometry", text)
        self.assertNotIn("CONFLICT", text)

    def test_unknown_units_never_yield_a_real_area(self) -> None:
        text = self.summarize(units=0)
        self.assertIn("unknown ($INSUNITS=0)", text)
        self.assertIn("not computed (unknown units)", text)
        self.assertNotIn(" m2 |", text)

    def test_garbage_is_a_malformed_stub(self) -> None:
        result = cad_extractors.extract_dxf(b"not a drawing")
        self.assertEqual((result.status, result.reason), ("stub", "malformed"))


@unittest.skipUnless(HAVE_IFC, "needs the bim extra (ifcopenshell)")
class TestIfc(unittest.TestCase):
    def test_reports_property_coverage_and_missing_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "doors.ifc"
            make_ifc(path)
            result = cad_extractors.extract_ifc(path.read_bytes())
        self.assertEqual(result.status, "ok")
        self.assertIn("schema IFC4", result.text)
        self.assertIn("Training building", result.text)
        self.assertIn("| Pset_DoorCommon | FireRating | 1/2 (1 missing) | 60min x1 |", result.text)
        self.assertIn("data-only", result.text)
        self.assertIn("| D02 |", result.text)

    def test_garbage_is_a_malformed_stub(self) -> None:
        result = cad_extractors.extract_ifc(b"not a model")
        self.assertEqual((result.status, result.reason), ("stub", "malformed"))


class TestDwg(unittest.TestCase):
    def test_release_is_read_from_magic_bytes(self) -> None:
        self.assertEqual(cad_extractors.dwg_release(b"AC1032\x00\x00"), "AC1032 (AutoCAD 2018)")
        self.assertIn("unrecognized", cad_extractors.dwg_release(b"XXXXXX"))

    @unittest.skipUnless(HAVE_EZDXF, "needs the cad extra (ezdxf)")
    def test_converted_drawing_is_summarized_with_a_conversion_caveat(self) -> None:
        def fake_convert(source: pathlib.Path, target: pathlib.Path) -> bool:
            make_plate(target)
            return True

        original = cad_extractors._convert_dwg
        cad_extractors._convert_dwg = fake_convert
        try:
            result = cad_extractors.extract_dwg(b"AC1032" + b"\x00" * 64)
        finally:
            cad_extractors._convert_dwg = original
        self.assertEqual(result.status, "ok")
        self.assertIn("# DWG drawing summary", result.text)
        self.assertIn("Native DWG release AC1032 (AutoCAD 2018)", result.text)
        self.assertIn("lower bound on the native file", result.text)

    def test_failed_conversion_is_a_stub_that_keeps_the_release(self) -> None:
        original = cad_extractors._convert_dwg
        cad_extractors._convert_dwg = lambda source, target: False
        try:
            result = cad_extractors.extract_dwg(b"AC1015" + b"\x00" * 64)
        finally:
            cad_extractors._convert_dwg = original
        self.assertEqual((result.status, result.reason), ("stub", "convert-failed"))
        self.assertIn("AC1015 (AutoCAD 2000)", result.text)


class TestWiring(unittest.TestCase):
    """Sidecar routing for CAD suffixes, independent of what's installed."""

    def setUp(self) -> None:
        extractors.clear_process_cache()
        extractors._WARNED_BACKENDS.clear()
        self._backends = dict(extractors._BACKENDS)

    def tearDown(self) -> None:
        extractors._BACKENDS.clear()
        extractors._BACKENDS.update(self._backends)
        extractors.clear_process_cache()

    def set_ready(self, suffix: str, ready: bool) -> None:
        extractors._BACKENDS[suffix] = dataclasses.replace(
            extractors._BACKENDS[suffix], probe=lambda: ready
        )

    def test_cad_files_are_discovered_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name in ("a.dxf", "b.dwg", "c.ifc", "d.ids"):
                (root / name).write_bytes(b"x")
            found = {p.name for p in discover_files(root)}
        self.assertEqual(found, {"a.dxf", "b.dwg", "c.ifc", "d.ids"})

    def test_ids_is_indexed_as_a_sidecar_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "doors.ids").write_bytes(IDS_XML)
            docs = load_documents(root)
            entry = extractors.load_manifest(root)["entries"]["doors.ids"]
        self.assertEqual([d.docid for d in docs], [".agentic-retrieval/extracted/doors.ids.md"])
        self.assertIn("FireRating", docs[0].text)
        self.assertEqual(entry["extractor_version"], "ids-xml/1")

    def test_missing_dwg_backend_writes_guidance_stub_with_release(self) -> None:
        self.set_ready(".dwg", False)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "plan.dwg").write_bytes(b"AC1027" + b"\x00" * 64)
            with redirect_stderr(StringIO()) as err:
                sidecar = extractors.ensure_sidecar(root, root / "plan.dwg")
        self.assertEqual((sidecar.status, sidecar.reason), ("stub", "backend-missing"))
        self.assertIn("dwg2dxf", sidecar.text)
        self.assertIn("AC1027 (AutoCAD 2013)", sidecar.text)
        self.assertIn("DWG converter is not available", err.getvalue())

    def test_stub_self_heals_once_backend_appears(self) -> None:
        self.set_ready(".ids", False)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "doors.ids").write_bytes(IDS_XML)
            with redirect_stderr(StringIO()):
                first = extractors.ensure_sidecar(root, root / "doors.ids")
            self.assertEqual(first.reason, "backend-missing")
            self.assertFalse(extractors.needs_reextraction(root))
            self.set_ready(".ids", True)
            extractors.clear_process_cache()
            self.assertTrue(extractors.needs_reextraction(root))
            healed = extractors.ensure_sidecar(root, root / "doors.ids")
        self.assertEqual((healed.status, healed.cache_hit), ("ok", False))

    def test_pdf_backend_state_does_not_leak_into_cad_suffixes(self) -> None:
        original = extractors.backend_available
        extractors.backend_available = lambda: False
        try:
            self.assertTrue(extractors.backend_ready(".ids"))
            self.assertFalse(extractors.backend_ready(".pdf"))
        finally:
            extractors.backend_available = original

    def test_agent_transcript_survives_for_a_cad_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "doors.ids").write_bytes(IDS_XML)
            extractors.register_sidecar(root, root / "doors.ids", "Hand-written summary.")
            sidecar = extractors.ensure_sidecar(root, root / "doors.ids")
        self.assertTrue(sidecar.cache_hit)
        self.assertIn("Hand-written summary.", sidecar.text)


if __name__ == "__main__":
    unittest.main()
