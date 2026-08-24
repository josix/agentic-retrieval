# Authoring Office/image sidecar transcripts

Authoritative guide for the agent-authored path for `.docx`, `.pptx`,
`.xlsx`, and images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) — the
formats with **no machine extractor at all**. See the `retrieval` skill's
"Author the media transcript yourself" section for the shared workflow
steps (detect/scope/read/write/register/reindex/verify); this reference
covers the Office/image-specific mechanics that section only summarizes.

## Entry point

`retrieval sidecar --list --root "$PROJECT_ROOT" --json` reports every
`.docx`/`.pptx`/`.xlsx`/image file as `state: stub`, `reason: agent-only`.
There is nothing to install, and no `--extra` flips this — `retrieval
extract` never touches these suffixes; `sidecar --register` is the only
way to replace the stub with real content.

## Read strategy

Use your own `Read()` tool directly on the original file first — most
coding-agent `Read` implementations render `.docx`/`.pptx`/`.xlsx` text
directly, and images are handled by native vision. If `Read()` returns
binary garbage, an error, or nothing usable, **leave the stub as-is** —
never guess a transcript from the filename alone.

## Heading conventions

Chunking treats any Markdown heading (`#` through `######`) as a section
break, and the heading text becomes the breadcrumb attached to every
chunk under it (`chunker.py`'s `_is_heading`/heading-capture logic feeds
`project_loader.py`'s `load_chunk_documents`, which sets each chunk's
`Document.context` to the enclosing heading). Use one H2 per logical unit:

| Format | Heading | Notes |
|---|---|---|
| `.docx` | `## Section: <the doc's own heading>` | Fall back to `## Part N` if the source has no headings of its own |
| `.pptx` | `## Slide N — <title>` | One heading per slide |
| `.xlsx` | `## Sheet: <name>` | Use `## Sheet: <name> (continued)` if a sheet needs more than one section |
| Images | `## Image: <filename>` or `## Figure: <caption>` | Whichever is more descriptive |

`## Page N` is reserved for genuinely paginated sources (PDFs). Don't
invent fake page numbers for these formats just to get a breadcrumb —
use the convention above instead.

## Section sizing

Sidecars are chunked like any other prose document, at this repo's
`prose_chars` target of **400 characters** (`chunker.py`'s
`ChunkingPolicy.prose_chars`), not the "300-600 tokens" guidance you may
have seen in general transcript-authoring advice — that figure is an
external best practice, not this repo's rule. Aim for one heading per
logical unit, sections roughly 200-1500 characters, blank-line-separated
paragraphs, and don't put a heading on every single bullet — group
related bullets under one heading instead.

## Per-format body guidance

- **`.pptx`**: bullet points for the slide's visible text, plus a
  `(notes)` line for speaker notes if present, plus a short prose
  description for any diagram or chart that isn't just text.
- **`.xlsx`**: describe the sheet's purpose, list column headers and
  units, and include a few representative rows — never dump every row of
  a large sheet.
- **`.docx`**: render tables as Markdown tables; accepted/tracked text
  goes in as plain prose.
- **Images**: describe visible text, structure, and meaning — what the
  image actually shows, not what you assume it shows.

## Anti-fabrication

Never write content you did not actually see. Use `_[illegible]_` for
text you can't make out and `_[no text content]_` for a section with
nothing to transcribe. If a tool didn't render the file at all, leave the
stub rather than inventing a transcript. For decorative images (logos,
icons), either leave the stub or register a single honest line — don't
manufacture detail to fill space. Flag low-confidence transcriptions
inline (e.g. "likely reads ... but partially obscured") rather than
stating them as fact.

## Mechanics

- Never write a `<!-- source: ... -->` header yourself — `retrieval
  sidecar --register` adds it for you.
- Never hand-edit a sidecar `.md` file under `.agentic-retrieval/extracted/`
  — its byte size is part of the cache-hit check (`extractors.py`'s
  `_is_cache_hit`), so an out-of-band edit silently turns it into a cache
  miss and it gets overwritten on the next extraction/reindex. Always go
  through `register_sidecar` (i.e. `retrieval sidecar --register`).
- Register every transcript you've authored before reindexing — the
  manifest write is last-writer-wins per file, not additive.
- Re-register (repeat the read/write/register steps) whenever the source
  file itself changes.

## Known nuances

- The manifest's `pages` counter only counts `## Page N` headings, so it
  reads `0` for every Office/image sidecar regardless of how well it's
  sectioned with the headings above — this is cosmetic (as of engine
  0.10.0) and does not affect breadcrumbs, which come from any `## `
  heading, not specifically `## Page N`.
- A source file over `EXTRACT_MAX_BYTES` (25 MB) never appears in
  `sidecar --list` at all — it's excluded at discovery time. Pass an
  `extract_max_bytes` override to `discover_files`/`load_documents` (see
  [Customize what gets indexed](../../../docs/how-to/customize-indexing.md))
  if you need to raise the cap for a large corpus.

## Verification loop

1. `retrieval sidecar --register <file> --transcript <file> --root
   "$PROJECT_ROOT"`.
2. `retrieval index` (or a bare `retrieval query`, which auto-reindexes).
3. `retrieval sidecar --list` should show the file as `agent-authored`.
4. Query a distinctive phrase from the transcript — it should hit
   `.agentic-retrieval/extracted/<rel>.md` with the breadcrumb you expect
   (e.g. `Slide 3 — ...`, `Sheet: Q3`).
