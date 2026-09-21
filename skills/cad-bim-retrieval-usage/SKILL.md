---
name: cad-bim-retrieval-usage
description: This skill should be used when the user asks to find, fetch, or explain CAD/BIM files by description — DWG or DXF drawings, IFC building models, IDS requirement files — e.g. "which drawing has the fire-rated doors", "find the plate with four holes", "what's in this .dwg", "find the IFC missing door properties", or when handling a DWG/DXF/IFC backend-missing stub.
---

# CAD/BIM Retrieval Usage (DWG, DXF, IFC, IDS)

Find the CAD/BIM file that matches a description, then explain what the
parser actually found in it. Built on the same sidecar-transcript mechanism
as PDFs (`retrieval/extractors.py`), with format parsers in
`retrieval/cad_extractors.py`.

## What it is

A drawing or model has no prose to search, so each file is parsed once into
a descriptive Markdown sidecar at
`.agentic-retrieval/extracted/<relpath>.md`, and that sidecar is what gets
indexed and what search hits point at.

| Suffix | Backend | Sidecar describes |
|---|---|---|
| `.dxf` | `ezdxf` (`cad` extra) | units, layers with per-type entity counts and visibility, block references and attribute values, every text string verbatim, dimensions (measured vs. override text), circles grouped by diameter, closed-polyline areas, hole-callout vs. geometry conflicts, external dependencies (xrefs, images, underlays, fonts) |
| `.dwg` | `cad` extra **plus** a converter on `PATH`: LibreDWG `dwg2dxf` or the ODA File Converter | the same as DXF, plus the native DWG release and a conversion caveat |
| `.ifc` | `ifcopenshell` (`bim` extra) | schema, project, units, spatial structure, element counts by class, property-set coverage per class (who is missing what), elements with GlobalId and STEP id, materials, classification, whether any geometry exists |
| `.ids` | none (stdlib XML) | each specification in words: which elements it applies to, what each must satisfy, required / optional / prohibited |

Every number comes from a parser reading the file. Nothing is inferred from
a rendering, and nothing is guessed: unknown units mean no real-world area,
a boundary with arcs means no area, a missing property is reported as
missing.

## Setup

Add the extras to whatever `$EXTRAS` Step 1 of the `retrieval` skill already
syncs, and use the same `$EXTRAS` on every `uv run`:

```bash
EXTRAS="$EXTRAS --extra cad --extra bim"
```

`.ids` needs nothing. Native `.dwg` additionally needs a converter, which
is a system binary, not a Python package:

```bash
brew install libredwg        # provides dwg2dxf (GPL-3.0); or install the ODA File Converter
```

Check what this environment can parse before promising anything:

```bash
uv run --project "$ENGINE" $EXTRAS retrieval sidecar --list --root "$PROJECT_ROOT" --json
# -> "backends": {".dwg": false, ".dxf": true, ".ifc": true, ".ids": true, ".pdf": false}
```

## Workflow

1. **Index.** CAD/BIM files are discovered by default; `retrieval index`
   (or the first `query`) parses them. For a large drawing set, pre-warm
   with `retrieval extract --root "$PROJECT_ROOT"` — it prints one line per
   file, `ok` with a character count or `stub (<reason>)`.
2. **Query by description**, in the user's own words:

   ```bash
   retrieval query "plate where the hole note disagrees with the drilled holes" \
     --root "$PROJECT_ROOT" --top-k 5
   ```

   A hit looks like
   `.agentic-retrieval/extracted/drawings/plate.dxf.md:44-46 ... Hole callouts versus geometry`.
   The sidecar path minus the `.agentic-retrieval/extracted/` prefix and the
   trailing `.md` is the original file: `drawings/plate.dxf`.
3. **Read the whole sidecar**, not just the hit span. A summary is short
   (usually under 100 lines) and its sections qualify each other — the
   Caveats section decides how much of the Geometry section you may repeat.
4. **Answer with the file and the explanation together** (see below).

## Presenting the answer

Give the user, in this order:

- **Which file(s)** matched, as project-relative paths to the *original*
  file, best match first. If two files are near-duplicates (revisions
  `plan_A` / `plan_B`), say so and say what differs between their sidecars.
- **What is in it**, in plain language, drawn from the sidecar: what kind of
  file, units, what it contains, the text it carries.
- **Evidence** the user can check: layer names, entity handles, GlobalIds,
  STEP ids — copy them from the sidecar tables.
- **What the parser could not establish**, from the Caveats section and any
  "not computed" cell. Surface a `CONFLICT` line or a coverage gap
  (`2/3 (1 missing)`) as a question for the author, not as a verdict.

Do not upgrade the sidecar's claims. It reports that a door's `FireRating`
is the string `60min`; it does not say the door complies with any code. It
reports a closed polyline of 12 m² on layer `A-ROOM`; calling that "a room"
is your inference and must be labelled as one.

## When a file comes back as a stub

`retrieval sidecar --list` shows `stub (backend-missing)`, or the hit body
is a guidance paragraph instead of a summary.

- **Install the backend** named in the stub (extras above, or the DWG
  converter) and reindex; stubs self-heal on the next `index`/`query` once
  the backend is present.
- **If it cannot be installed**, do not pretend to read the file. You cannot
  usefully `Read()` a DWG (binary) or a raw DXF/IFC (thousands of low-level
  lines). A DWG stub still reports the AutoCAD release from the file header
  — tell the user that, and that the content is unread. If another tool can
  summarize the file (a CAD export, a vendor viewer's report), register its
  output with `retrieval sidecar --register <file> --transcript <summary>`.
- `stub (malformed)` / `stub (convert-failed)` means the parser or converter
  ran and failed on this file. Report it as unreadable; the converter
  supporting a release does not mean it supports every object in it.

## Limits

- Model space only; paper-space layouts are named, not summarized.
- A DWG summary describes the *converted* DXF. Converters drop or
  approximate objects they do not support, so entity counts are a lower
  bound on the native file, and handles refer to the converted file.
- Tables cap long lists (400 text rows, 60 geometry rows, 40 elements per
  IFC class) and say how many rows were omitted.
- An IDS summary describes requirements; it validates nothing. For pass/fail
  against a model, run an IDS validator (e.g. `ifctester`) — and note that
  validators themselves differ on edge cases such as floating-point
  tolerance.
- Licenses: `ezdxf` is MIT, `ifcopenshell` is LGPL-3.0, LibreDWG is GPL-3.0
  and is invoked as a separate program, never linked.
