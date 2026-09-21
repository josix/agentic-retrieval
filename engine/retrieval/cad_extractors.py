"""CAD/BIM sidecar extractors: DXF, DWG, IFC and IDS -> descriptive Markdown.

Imported lazily by ``retrieval.extractors`` (which stays stdlib-only at
import scope); this module is itself stdlib-only at import scope and imports
``ezdxf`` / ``ifcopenshell`` only inside the functions that need them, the
same guidance convention ``extractors._extract_pdf`` follows for ``pypdf``.

A CAD/BIM file has no prose to transcribe, so each extractor writes a
*description* of what was parsed: counts, names, text content, measured
values and explicit caveats, phrased so both lexical and dense retrievers can
match a natural-language request ("the plan with the fire-rated doors", "the
plate with four holes") and so the agent can cite entity handles / GlobalIds.
Every number is read from the file by a parser; nothing is inferred from a
rendering. Where a value cannot be established (unknown units, arcs in a
boundary, missing property) the transcript says so instead of guessing.

Backends:

- ``.dxf`` -> ``ezdxf`` (``cad`` extra), tolerant ``recover`` loader.
- ``.dwg`` -> an external converter (LibreDWG ``dwg2dxf`` or the ODA File
  Converter) plus ``ezdxf``. No pure-Python DWG reader exists; without a
  converter the file stays a ``backend-missing`` stub that still reports the
  DWG release read from the file header.
- ``.ifc`` -> ``ifcopenshell`` (``bim`` extra).
- ``.ids`` -> stdlib ``xml.etree`` (no backend; IDS is plain XML).
"""

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from retrieval.extractors import MAX_TRANSCRIPT_CHARS, Extraction

DXF_VERSION = "ezdxf-summary/1"
DWG_VERSION = "dwg-convert+ezdxf-summary/1"
IFC_VERSION = "ifcopenshell-summary/1"
IDS_VERSION = "ids-xml/1"

#: Per-section row caps: a transcript is a retrieval target, not a dump. The
#: section says how many rows were omitted so the agent knows to open the
#: source with a real tool when it needs the long tail.
_MAX_TEXT_ROWS = 400
_MAX_GEOMETRY_ROWS = 60
_MAX_ELEMENT_ROWS = 40
_MAX_VALUES = 6

_DWG_CONVERT_TIMEOUT_S = 120

#: ``$INSUNITS`` code -> (label, metres per drawing unit). Codes outside this
#: table are reported as unknown rather than mapped to a guess.
_INSUNITS = {
    1: ("inches", 0.0254), 2: ("feet", 0.3048), 4: ("millimeters", 0.001),
    5: ("centimeters", 0.01), 6: ("meters", 1.0), 7: ("kilometers", 1000.0),
    10: ("yards", 0.9144), 14: ("decimeters", 0.1),
}

#: First six bytes of a DWG file -> AutoCAD release.
_DWG_RELEASES = {
    b"AC1009": "R11/R12", b"AC1012": "R13", b"AC1014": "R14", b"AC1015": "2000",
    b"AC1018": "2004", b"AC1021": "2007", b"AC1024": "2010", b"AC1027": "2013",
    b"AC1032": "2018",
}

#: "4X DIA 8", "2x Ø6.5", "4X %%c8" style hole callouts.
_HOLE_NOTE_RE = re.compile(
    r"(\d+)\s*[xX×]\s*(?:DIA\.?|DIAMETER|Ø|⌀|%%[cC])\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)

_MSG_DXF_MALFORMED = "This DXF could not be parsed; it may be corrupted or not a DXF file."
_MSG_IFC_MALFORMED = "This IFC file could not be parsed; it may be corrupted or not IFC STEP."
_MSG_IDS_MALFORMED = "This IDS file could not be parsed as XML."
_MSG_DWG_CONVERT_FAILED = (
    "The DWG converter ran but produced no readable DXF for this file, so its "
    "content could not be summarized."
)


def _stub(reason: str, message: str) -> Extraction:
    return Extraction(text=message, status="stub", reason=reason, pages=0, truncated=False)


def _ok(sections: Iterable[str]) -> Extraction:
    text = "\n\n".join(s for s in sections if s).strip() + "\n"
    truncated = len(text) > MAX_TRANSCRIPT_CHARS
    if truncated:
        marker = "\n\n_[transcript truncated: exceeded extraction size budget]_"
        text = text[: MAX_TRANSCRIPT_CHARS - len(marker)] + marker
    return Extraction(text=text, status="ok", reason="", pages=0, truncated=truncated)


def _num(value: float) -> str:
    """Compact, stable number rendering (``6`` not ``6.0``, 4 significant decimals)."""
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _cell(value: Any) -> str:
    """Make *value* safe inside a one-line Markdown table cell."""
    return re.sub(r"\s+", " ", str(value)).replace("|", "\\|").strip()


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _omitted(total: int, shown: int, what: str) -> str:
    return f"\n\n_{total - shown} more {what} omitted; open the source file for the full list._" \
        if total > shown else ""


# --------------------------------------------------------------------- DXF --


def dwg_converter() -> str | None:
    """Name of the first usable DWG->DXF converter, or ``None``."""
    if shutil.which("dwg2dxf"):
        return "dwg2dxf"
    try:
        from ezdxf.addons import odafc
    except ImportError:
        return None
    return "odafc" if odafc.is_installed() else None


def dwg_release(data: bytes) -> str:
    """AutoCAD release named by a DWG's magic bytes, e.g. ``AC1032 (AutoCAD 2018)``."""
    magic = data[:6]
    code = magic.decode("ascii", "replace")
    release = _DWG_RELEASES.get(magic)
    return f"{code} (AutoCAD {release})" if release else f"{code} (unrecognized release)"


def _units(doc: Any) -> Tuple[str, float | None]:
    code = doc.header.get("$INSUNITS", 0)
    if code in _INSUNITS:
        label, scale = _INSUNITS[code]
        return f"{label} ($INSUNITS={code})", scale
    return f"unknown ($INSUNITS={code})", None


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
                   for i in range(n))) / 2


def _entity_text(entity: Any) -> str:
    kind = entity.dxftype()
    if kind == "MTEXT":
        return entity.plain_text()
    if kind in ("TEXT", "ATTRIB", "ATTDEF"):
        return entity.dxf.get("text", "")
    return ""


def _dxf_layers(doc: Any, msp: Any) -> str:
    by_layer: Dict[str, Counter] = defaultdict(Counter)
    for e in msp:
        by_layer[e.dxf.get("layer", "0")][e.dxftype()] += 1
    rows = []
    for layer in doc.layers:
        name = layer.dxf.name
        kinds = by_layer.pop(name, Counter())
        state = ", ".join(s for s, on in [("off", layer.is_off()), ("frozen", layer.is_frozen()),
                                          ("locked", layer.is_locked())] if on) or "visible"
        rows.append([name, sum(kinds.values()),
                     ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) or "-", state])
    rows += [[name, sum(k.values()), ", ".join(f"{t} {n}" for t, n in sorted(k.items())),
              "not in layer table"] for name, k in sorted(by_layer.items())]
    note = ("Entity counts are per layer in model space. A quantity take-off must state which "
            "layers it includes; hidden, construction and annotation layers are listed, not "
            "filtered.")
    return "## Layers\n\n" + note + "\n\n" + _table(["Layer", "Entities", "Types", "State"], rows)


def _dxf_blocks(doc: Any, msp: Any) -> str:
    inserts = Counter(e.dxf.name for e in msp.query("INSERT"))
    rows = []
    for name, count in sorted(inserts.items()):
        attribs = Counter()
        for e in msp.query("INSERT"):
            if e.dxf.name == name:
                attribs.update(f"{a.dxf.tag}={a.dxf.text}" for a in e.attribs)
        sample = "; ".join(v for v, _ in attribs.most_common(_MAX_VALUES)) or "-"
        rows.append([name, count, sample])
    if not rows:
        return ""
    return ("## Block references\n\nNamed blocks inserted in model space (symbols such as doors, "
            "fixtures, title blocks) and a sample of their attribute values.\n\n"
            + _table(["Block", "Inserts", "Attribute values"], rows))


def _dxf_texts(msp: Any) -> Tuple[str, List[str]]:
    rows, plain = [], []
    for e in msp:
        text = _entity_text(e).strip()
        if e.dxftype() == "INSERT":
            for a in e.attribs:
                if a.dxf.text.strip():
                    rows.append([f"{a.dxf.tag}: {a.dxf.text}", "ATTRIB", a.dxf.get("layer", "0"),
                                 a.dxf.handle])
        if text:
            rows.append([text, e.dxftype(), e.dxf.get("layer", "0"), e.dxf.handle])
            plain.append(text)
    if not rows:
        return "", plain
    shown = rows[:_MAX_TEXT_ROWS]
    body = _table(["Text", "Type", "Layer", "Handle"], shown)
    return ("## Text and annotations\n\nEvery text string in model space, verbatim.\n\n" + body
            + _omitted(len(rows), len(shown), "text entities")), plain


def _dxf_dimensions(msp: Any) -> str:
    rows = []
    for e in msp.query("DIMENSION"):
        try:
            measured = e.get_measurement()
            # Ordinate dimensions return a point, not a length: no scalar to report.
            measured = _num(measured) if isinstance(measured, (int, float)) else "n/a (ordinate)"
        except Exception:  # noqa: BLE001 - malformed dimension geometry
            measured = "n/a"
        override = e.dxf.get("text", "")
        shown = "measured value" if override in ("", "<>") else f"override text {override!r}"
        rows.append([measured, shown, e.dxf.get("layer", "0"), e.dxf.handle])
    if not rows:
        return ""
    shown_rows = rows[:_MAX_GEOMETRY_ROWS]
    return ("## Dimensions\n\nMeasured value comes from the dimension's geometry; an override "
            "text replaces what the drawing displays and can disagree with the geometry.\n\n"
            + _table(["Measured", "Displays", "Layer", "Handle"], shown_rows)
            + _omitted(len(rows), len(shown_rows), "dimensions"))


def _dxf_geometry(msp: Any, unit_label: str, scale: float | None) -> Tuple[str, Counter]:
    circles: Counter = Counter()
    for e in msp.query("CIRCLE"):
        circles[(e.dxf.get("layer", "0"), round(2 * e.dxf.radius, 6))] += 1
    parts = []
    if circles:
        rows = [[layer, _num(dia), n] for (layer, dia), n in sorted(circles.items())]
        parts.append("Circles grouped by layer and diameter (drawing units):\n\n"
                     + _table(["Layer", "Diameter", "Count"], rows))
    polys = []
    for e in msp.query("LWPOLYLINE"):
        if not e.closed:
            continue
        if any(abs(p[4]) > 0 for p in e.get_points()):
            polys.append([e.dxf.get("layer", "0"), "not computed (boundary contains arcs)", "-",
                          len(e), e.dxf.handle, -1.0])
            continue
        area = _polygon_area([(p[0], p[1]) for p in e.get_points("xy")])
        real = f"{_num(area * scale ** 2)} m2" if scale else "not computed (unknown units)"
        polys.append([e.dxf.get("layer", "0"), _num(area), real, len(e), e.dxf.handle, area])
    if polys:
        polys.sort(key=lambda r: -r[5])
        shown = [r[:5] for r in polys[:_MAX_GEOMETRY_ROWS]]
        parts.append("Closed polylines (candidate rooms, profiles or regions), largest first:\n\n"
                     + _table(["Layer", "Area (drawing units2)", "Area (real)", "Vertices",
                               "Handle"], shown)
                     + _omitted(len(polys), len(shown), "closed polylines"))
    if not parts:
        return "", circles
    return f"## Geometry\n\nDrawing units: {unit_label}.\n\n" + "\n\n".join(parts), circles


def _dxf_hole_notes(texts: List[str], circles: Counter) -> str:
    by_diameter: Counter = Counter()
    for (_, dia), n in circles.items():
        by_diameter[dia] += n
    findings = []
    for text in texts:
        for count, dia in _HOLE_NOTE_RE.findall(text):
            wanted, diameter = int(count), float(dia)
            actual = by_diameter.get(round(diameter, 6), 0)
            if actual == wanted:
                findings.append(f"- Note {text!r} matches the geometry: {actual} circle(s) of "
                                f"diameter {_num(diameter)}.")
            else:
                present = ", ".join(f"{n} x diameter {_num(d)}" for d, n in sorted(
                    by_diameter.items())) or "no circles"
                findings.append(f"- CONFLICT: note {text!r} calls for {wanted} hole(s) of diameter "
                                f"{_num(diameter)}, but the geometry has {actual}; circles "
                                f"present: {present}. Neither value is authoritative - ask the "
                                "author.")
    if not findings:
        return ""
    return "## Hole callouts versus geometry\n\n" + "\n".join(findings)


def _dxf_dependencies(doc: Any) -> str:
    rows = []
    for block in doc.blocks:
        head = block.block
        if head is not None and (head.is_xref or head.is_xref_overlay):
            rows.append(["xref", head.dxf.get("xref_path", "") or block.name])
    rows += [["image", d.dxf.get("filename", "")] for d in doc.objects.query("IMAGEDEF")]
    rows += [["underlay", d.dxf.get("filename", "")]
             for d in doc.objects.query("PDFDEFINITION DWFDEFINITION DGNDEFINITION")]
    fonts = sorted({s.dxf.get("font", "") for s in doc.styles} - {""})
    rows += [["font", f] for f in fonts]
    if not rows:
        return ""
    return ("## External dependencies\n\nFiles this drawing references. Whether they exist next "
            "to the drawing was not checked; a missing xref or font changes what a recipient "
            "sees.\n\n" + _table(["Kind", "Path"], rows))


def _summarize_dxf(path: Path, origin: str) -> Extraction:
    import ezdxf
    from ezdxf import recover

    try:
        doc, auditor = recover.readfile(str(path))
    except (IOError, ezdxf.DXFError, UnicodeError, ValueError):
        return _stub("malformed", _MSG_DXF_MALFORMED)
    msp = doc.modelspace()
    unit_label, scale = _units(doc)
    kinds = Counter(e.dxftype() for e in msp)
    layouts = [name for name in doc.layout_names() if name != "Model"]
    texts_section, texts = _dxf_texts(msp)
    geometry_section, circles = _dxf_geometry(msp, unit_label, scale)

    caveats = []
    if scale is None:
        caveats.append("Units are unknown, so no real-world length or area is reported. Do not "
                       "assume millimeters.")
    if auditor.errors or auditor.fixes:
        caveats.append(f"The tolerant loader reported {len(auditor.errors)} unrecoverable "
                       f"error(s) and applied {len(auditor.fixes)} automatic fix(es).")
    caveats.append("Only model space is summarized. Material, thickness, tolerances and design "
                   "intent are not in the geometry unless a text entity states them.")

    overview = (
        f"# {origin} drawing summary\n\n"
        f"{origin} drawing, format {doc.dxfversion} (AutoCAD {doc.acad_release}). "
        f"Drawing units: {unit_label}. Model space holds {len(msp)} entities on "
        f"{len(doc.layers)} layers"
        + (f": {', '.join(f'{n} {k}' for k, n in kinds.most_common())}." if kinds else ".")
        + (f" Paper-space layouts: {', '.join(layouts)}." if layouts else "")
    )
    return _ok([overview, _dxf_layers(doc, msp), _dxf_blocks(doc, msp), texts_section,
                _dxf_dimensions(msp), geometry_section, _dxf_hole_notes(texts, circles),
                _dxf_dependencies(doc), "## Caveats\n\n" + "\n".join(f"- {c}" for c in caveats)])


def extract_dxf(data: bytes) -> Extraction:
    """Summarize DXF *data* (ASCII or binary) via ``ezdxf``'s tolerant loader."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.dxf"
        path.write_bytes(data)
        return _summarize_dxf(path, "DXF")


def _convert_dwg(source: Path, target: Path) -> bool:
    converter = dwg_converter()
    try:
        if converter == "dwg2dxf":
            subprocess.run(["dwg2dxf", "-y", "-o", str(target), str(source)], check=False,
                           capture_output=True, timeout=_DWG_CONVERT_TIMEOUT_S)
        elif converter == "odafc":
            from ezdxf.addons import odafc
            odafc.convert(str(source), str(target), replace=True)
        else:
            return False
    except (OSError, subprocess.SubprocessError, Exception):  # noqa: BLE001 - odafc raises broadly
        return False
    return target.is_file() and target.stat().st_size > 0


def extract_dwg(data: bytes) -> Extraction:
    """Convert DWG *data* to DXF with an external converter, then summarize it."""
    release = dwg_release(data)
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "source.dwg", Path(tmp) / "source.dxf"
        source.write_bytes(data)
        if not _convert_dwg(source, target):
            return _stub("convert-failed", f"{_MSG_DWG_CONVERT_FAILED} DWG release: {release}.")
        result = _summarize_dxf(target, "DWG")
    if result.status != "ok":
        return _stub("convert-failed", f"{_MSG_DWG_CONVERT_FAILED} DWG release: {release}.")
    note = (f"\n## Conversion\n\nNative DWG release {release}, converted to DXF by "
            f"{dwg_converter()} before parsing. Entity handles refer to the converted DXF; a "
            "converter can drop or approximate objects it does not support, so counts are a "
            "lower bound on the native file.\n")
    return Extraction(text=result.text + note, status="ok", reason="", pages=0,
                      truncated=result.truncated)


# --------------------------------------------------------------------- IFC --


def _ifc_value_summary(values: List[Any]) -> str:
    counts = Counter("(empty)" if v in (None, "") else str(v) for v in values)
    return ", ".join(f"{v} x{n}" for v, n in counts.most_common(_MAX_VALUES)) \
        + (" ..." if len(counts) > _MAX_VALUES else "")


def _ifc_psets(model: Any, products: List[Any]) -> str:
    import ifcopenshell.util.element as element_util

    by_class: Dict[str, List[Any]] = defaultdict(list)
    for p in products:
        by_class[p.is_a()].append(p)
    blocks = []
    for ifc_class, items in sorted(by_class.items()):
        props: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for item in items:
            for pset, values in element_util.get_psets(item).items():
                for name, value in values.items():
                    if name != "id":
                        props[(pset, name)].append(value)
        if not props:
            blocks.append(f"### {ifc_class} ({len(items)})\n\nNo property sets on any instance.")
            continue
        rows = []
        for (pset, name), values in sorted(props.items()):
            missing = len(items) - len(values)
            coverage = f"{len(values)}/{len(items)}" + (f" ({missing} missing)" if missing else "")
            rows.append([pset, name, coverage, _ifc_value_summary(values)])
        blocks.append(f"### {ifc_class} ({len(items)})\n\n"
                      + _table(["Property set", "Property", "Coverage", "Values"], rows))
    if not blocks:
        return ""
    return ("## Property sets by class\n\nCoverage is instances carrying the property out of all "
            "instances of the class; a gap is a data-quality finding, not a code violation.\n\n"
            + "\n\n".join(blocks))


def _ifc_elements(products: List[Any]) -> str:
    import ifcopenshell.util.element as element_util

    by_class: Dict[str, List[Any]] = defaultdict(list)
    for p in products:
        by_class[p.is_a()].append(p)
    blocks = []
    for ifc_class, items in sorted(by_class.items()):
        rows = []
        for item in items[:_MAX_ELEMENT_ROWS]:
            container = element_util.get_container(item)
            rows.append([item.Name or "(unnamed)", item.GlobalId,
                         container.Name if container is not None else "-", f"#{item.id()}"])
        blocks.append(f"### {ifc_class}\n\n"
                      + _table(["Name", "GlobalId", "Contained in", "STEP id"], rows)
                      + _omitted(len(items), len(rows), f"{ifc_class} instances"))
    return "## Elements\n\n" + "\n\n".join(blocks) if blocks else ""


def _ifc_units(model: Any) -> str:
    rows = []
    for assignment in model.by_type("IfcUnitAssignment"):
        for unit in assignment.Units:
            if unit.is_a("IfcSIUnit"):
                rows.append(f"{unit.UnitType}: {(unit.Prefix or '')}{unit.Name}")
            elif unit.is_a("IfcConversionBasedUnit"):
                rows.append(f"{unit.UnitType}: {unit.Name}")
    return ", ".join(sorted(set(rows))) or "none declared"


def extract_ifc(data: bytes) -> Extraction:
    """Summarize IFC *data* (STEP physical file) via ``ifcopenshell``."""
    import ifcopenshell

    # Reject non-STEP input up front: ifcopenshell's failed-open path leaves a
    # half-built file object whose __del__ prints a traceback to stderr.
    if b"ISO-10303-21" not in data[:256]:
        return _stub("malformed", _MSG_IFC_MALFORMED)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.ifc"
        path.write_bytes(data)
        try:
            model = ifcopenshell.open(str(path))
        except Exception:  # noqa: BLE001 - ifcopenshell raises bare Error/RuntimeError types
            return _stub("malformed", _MSG_IFC_MALFORMED)
        projects = model.by_type("IfcProject")
        products = [p for p in model.by_type("IfcElement")]
        spatial = [s for s in model.by_type("IfcSpatialStructureElement")]
        with_geometry = sum(1 for p in products if p.Representation is not None)
        classes = Counter(p.is_a() for p in products)
        materials = sorted({m.Name for m in model.by_type("IfcMaterial") if m.Name})
        classifications = sorted({f"{c.Identification if hasattr(c, 'Identification') else ''} "
                                  f"{c.Name or ''}".strip()
                                  for c in model.by_type("IfcClassificationReference")} - {""})

        overview = (
            f"# IFC model summary\n\nIFC building model, schema {model.schema}. Project: "
            f"{', '.join(p.Name or '(unnamed)' for p in projects) or '(no IfcProject)'}. "
            f"Units: {_ifc_units(model)}. The model has {len(products)} building elements"
            + (f" ({', '.join(f'{n} {c}' for c, n in classes.most_common())})" if classes else "")
            + f" and {len(spatial)} spatial containers. {with_geometry} of {len(products)} "
            "elements carry a geometric representation"
            + (" - this file is data-only, so no quantity or clash can be derived from it."
               if products and not with_geometry else ".")
        )
        spatial_rows = [[s.is_a(), s.Name or "(unnamed)", s.GlobalId] for s in spatial]
        spatial_section = ("## Spatial structure\n\n"
                           + _table(["Class", "Name", "GlobalId"], spatial_rows)) \
            if spatial_rows else ""
        extras = []
        if materials:
            extras.append("Materials: " + ", ".join(materials[:40]) + ".")
        if classifications:
            extras.append("Classification references: " + ", ".join(classifications[:40]) + ".")
        caveats = ("## Caveats\n\n- Property values are reported as stored; a string such as "
                   "`60min` is not normalized or checked against any building code.\n- Use an "
                   "IDS file with a validator for pass/fail requirements; this summary only "
                   "describes what is present.")
        return _ok([overview, spatial_section, _ifc_psets(model, products),
                    _ifc_elements(products),
                    "## Materials and classification\n\n" + "\n\n".join(extras) if extras else "",
                    caveats])


# --------------------------------------------------------------------- IDS --

_IDS_NS = "{http://standards.buildingsmart.org/IDS}"
_XS_NS = "{http://www.w3.org/2001/XMLSchema}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ids_value(node: ET.Element | None) -> str:
    """Render an IDS value node: a ``simpleValue`` or an ``xs:restriction``."""
    if node is None:
        return "any value"
    simple = node.find(f"{_IDS_NS}simpleValue")
    if simple is not None:
        return f"`{(simple.text or '').strip()}`"
    restriction = node.find(f"{_XS_NS}restriction")
    if restriction is None:
        return "any value"
    parts = []
    enums = [c.get("value", "") for c in restriction.findall(f"{_XS_NS}enumeration")]
    if enums:
        parts.append("one of " + ", ".join(f"`{v}`" for v in enums))
    for child in restriction:
        kind = _local(child.tag)
        if kind != "enumeration":
            parts.append(f"{kind} `{child.get('value', '')}`")
    return " and ".join(parts) or "any value"


def _ids_facet(facet: ET.Element) -> str:
    kind = _local(facet.tag)
    fields = [f"{_local(child.tag)} {_ids_value(child)}" for child in facet]
    cardinality = facet.get("cardinality")
    data_type = facet.get("dataType")
    suffix = "".join([f" [{cardinality}]" if cardinality else "",
                      f" (dataType {data_type})" if data_type else ""])
    return f"- {kind}: " + ("; ".join(fields) or "any") + suffix


def _ids_occurs(applicability: ET.Element | None) -> str:
    if applicability is None:
        return "unspecified"
    lo, hi = applicability.get("minOccurs", "1"), applicability.get("maxOccurs", "unbounded")
    if hi == "0":
        return "prohibited (no matching element may exist)"
    return "required (at least one matching element)" if lo != "0" \
        else "optional (checked only if matching elements exist)"


def extract_ids(data: bytes) -> Extraction:
    """Describe an IDS (Information Delivery Specification) XML file."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return _stub("malformed", _MSG_IDS_MALFORMED)
    info = root.find(f"{_IDS_NS}info")
    meta = {_local(c.tag): (c.text or "").strip() for c in info} if info is not None else {}
    specs = root.findall(f"{_IDS_NS}specifications/{_IDS_NS}specification")
    overview = (
        f"# IDS requirements summary\n\nInformation Delivery Specification "
        f"{meta.get('title') or '(untitled)'!r} with {len(specs)} specification(s): machine-"
        "checkable information requirements for IFC models."
        + "".join(f" {k.capitalize()}: {v}." for k, v in meta.items() if k != "title" and v)
    )
    sections = [overview]
    for i, spec in enumerate(specs, start=1):
        applicability = spec.find(f"{_IDS_NS}applicability")
        requirements = spec.find(f"{_IDS_NS}requirements")
        lines = [f"## Specification {i}: {spec.get('name', '(unnamed)')}",
                 f"IFC versions: {spec.get('ifcVersion', 'unspecified')}. "
                 f"Occurrence: {_ids_occurs(applicability)}."]
        for label in ("description", "instructions"):
            if spec.get(label):
                lines.append(f"{label.capitalize()}: {spec.get(label)}")
        lines.append("Applies to elements matching:\n"
                     + ("\n".join(_ids_facet(f) for f in applicability)
                        if applicability is not None and len(applicability) else "- everything"))
        has_requirements = requirements is not None and len(requirements)
        lines.append("Each must satisfy:\n"
                     + ("\n".join(_ids_facet(f) for f in requirements)
                        if has_requirements else "- (no requirements)"))
        sections.append("\n\n".join(lines))
    sections.append("## Caveats\n\n- This describes the requirements; it does not check any "
                    "model. Run an IDS validator (e.g. ifctester) against an IFC file for "
                    "pass/fail results.")
    return _ok(sections)
