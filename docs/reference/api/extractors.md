# `retrieval.extractors`

Sidecar-transcript extraction for non-text-native document formats (PDF +
agent-only media). Stdlib-only at import scope — this module must import
cleanly with zero optional extras installed, same guarantee as the rest of
the default pipeline; the `pypdf` backend is imported lazily, only inside
the functions that actually need it.

`EXTRACTABLE_EXTENSIONS` splits into three tiers:

- `MACHINE_EXTRACTABLE_EXTENSIONS` (`.pdf`, `.srt`, `.vtt`) — has a
  registered machine extractor. `.pdf` needs the optional `pypdf` backend
  (see `PYPDF_EXTENSIONS`); `.srt`/`.vtt` (caption files) are extracted by a
  stdlib-only converter (`_extract_captions`, `captions/1`) that needs no
  extra at all.
- `AGENT_ONLY_EXTENSIONS` (`.docx`, `.pptx`, `.xlsx`, `.png`, `.jpg`,
  `.jpeg`, `.gif`, `.webp`) — no machine extractor exists at all, but a
  coding agent can read/view the file natively; every such file always
  indexes as an `"agent-only"`-reason stub, and `retrieval sidecar
  --register` is the only way to index its real content.
- `AGENT_ORCHESTRATED_EXTENSIONS` (`.mp4`, `.mov`, `.mkv`, `.webm`, `.mp3`,
  `.m4a`, `.wav`, `.flac`) — no machine extractor, and no native
  agent-readable path either: an agent must orchestrate an external ASR
  tool via Bash, then register the result. Every such file indexes as an
  `"agent-orchestrated"`-reason stub. Uncapped at discovery (see
  `project_loader._is_eligible_file`) and never byte-read by this module
  (see "Stat-only identity for tier 3" below).

`PYPDF_EXTENSIONS` (`{".pdf"}`) is the strict subset of
`MACHINE_EXTRACTABLE_EXTENSIONS` that actually needs the `pypdf` backend —
`require_extractors()` and `ensure_sidecar`'s backend-missing-stub branch
key off this set, not the broader one, so a caption-only corpus never
triggers the pypdf guidance error.

A file's extracted text is written once to a Markdown "sidecar" file under
`<project-root>/.agentic-retrieval/extracted/<rel-path>.md` (always under
the project root, deliberately ignoring `RETRIEVAL_INDEX_DIR` — a sidecar is
a citation target a coding agent `Read()`s by project-relative path, so it
must live in the tree being indexed even when the *cache* itself is
redirected elsewhere). A content-hash + extractor-version manifest
(`manifest.json`, alongside the sidecars) makes re-extraction a no-op on
unchanged files across the multiple loader passes `index --auto` performs
per run.

Every failure mode (encrypted, malformed, empty, no-text-layer,
backend-missing, agent-only) still produces a non-empty, human-readable
stub sidecar — an empty sidecar would yield zero chunks and silently vanish
from every index.

Key public functions relevant to the agent-authored recovery path:
`register_sidecar` (write a sidecar from a hand-authored transcript, never
imports `pypdf`), `sidecar_states` (per-source-file state report backing
`retrieval sidecar --list`), and `agent_sidecar_revision` (extracts the
fingerprint-relevant revision string from a manifest entry, consumed by
`retrieval.persistence.compute_fingerprint`).

::: retrieval.extractors

## Extractor-version / status taxonomy

| `extractor_version` | Written by | `status` |
|---|---|---|
| `pypdf-text/1` (`EXTRACTOR_VERSION`) | `ensure_sidecar` (the `pypdf` backend) | `ok`, or `stub` (see the stub taxonomy below) |
| `captions/1` (`CAPTIONS_EXTRACTOR_VERSION`) | `ensure_sidecar` (`.srt`/`.vtt`, stdlib-only) | `ok`, or `stub` (`no-cues`/`malformed`) |
| `agent-authored/1` (`AGENT_EXTRACTOR_VERSION`) | `register_sidecar` (an agent-authored transcript) | always `ok` — an agent transcript never renders as a stub |

All three extractor-version strings are accepted as "fresh" by
`_is_cache_hit` (membership in `_ACCEPTED_EXTRACTOR_VERSIONS`, which is
`{EXTRACTOR_VERSION, AGENT_EXTRACTOR_VERSION} | set(_EXTRACTOR_VERSIONS.values())`
— rebuilt automatically whenever `register_extractor(..., version=...)`
registers a new per-suffix version, not equality against a single
constant). `extractor_version_for(path)` returns the version to stamp for a
given path's suffix: its `_EXTRACTOR_VERSIONS` override if one is
registered, else the top-level `EXTRACTOR_VERSION`.

## Stub taxonomy (`status == "stub"`)

| `reason` | When |
|---|---|
| `encrypted` | Password-protected PDF; the empty-password decrypt attempt failed |
| `crypto-unavailable` | Encrypted PDF using a method needing the optional `cryptography` package |
| `empty` | PDF with zero pages |
| `no-text-layer` | Scanned/image-only PDF (no OCR performed) |
| `malformed` | Corrupted or unsupported PDF structure |
| `backend-missing` | PDF, but `pypdf` is not installed |
| `no-cues` | Caption file (`.srt`/`.vtt`) parses but contains zero cues (timed text) |
| `malformed` (captions) | Caption file has no `-->` cue-timing lines at all — not recognizable as SRT/VTT |
| `agent-only` | `AGENT_ONLY_EXTENSIONS` suffix (docx/pptx/xlsx/image) — no machine extractor exists for this format at all, regardless of what's installed; distinct from `backend-missing` so it's never mistaken for a stale pypdf stub that self-heals once pypdf becomes available (`needs_reextraction`/`_is_cache_hit` key on the literal `"backend-missing"` string) |
| `agent-orchestrated` | `AGENT_ORCHESTRATED_EXTENSIONS` suffix (audio/video) — no machine extractor, and no native agent-readable path either; distinct from both `agent-only` and `backend-missing` for the same self-heal-avoidance reason |

## Backend-missing degradation vs. hard failure

`index`/`query` never hard-fail on a missing `pypdf` backend — a PDF
without a usable backend still indexes as a searchable, backend-missing
stub, with a single `warning: pypdf is not installed` line printed to
stderr per process (`_warn_backend_missing`). The `retrieval extract`
subcommand is the one place a missing backend *does* hard-fail, via an
explicit preflight `require_extractors()` call — see [CLI reference:
`extract`](../cli.md#extract). `require_extractors`'s guidance message and
the `backend-missing` stub's own text both point at `retrieval sidecar
--register` as the no-install alternative.

## Supersede policy: agent-authored entries are never auto-replaced

An agent-authored sidecar (`register_sidecar`) is treated as a durable,
first-class transcript — installing the `pdf` extra later does **not**
cause it to be silently regenerated by pypdf (`needs_reextraction` and
`_is_cache_hit`'s backend-missing check both stay pinned on the literal
`reason == "backend-missing"` string, which an agent entry never has). The
one deliberate escape hatch is `retrieval extract --force`, which prints
`warning: overwriting N agent-authored sidecar(s) with pypdf output` to
stderr before doing so.

## Agent-only media has no degradation path

Unlike PDF's `backend-missing` stub, an `agent-only` stub never
"self-heals": `require_extractors` and `backend_available()` both only ever
concern `pypdf`/`MACHINE_EXTRACTABLE_EXTENSIONS`, so there is no install
that turns an `agent-only` stub into a real transcript — `retrieval sidecar
--register` is the only path. `extract` never attempts these suffixes
either (see [CLI reference: `extract`](../cli.md#extract)); its `--prune`
keep-set still preserves a registered agent-only sidecar, since pruning is
based on discovery (`extractors.EXTRACTABLE_EXTENSIONS`), not on what
`extract`'s own extraction loop touched.

## Stat-only identity for tier 3 (audio/video)

Every tier except `AGENT_ORCHESTRATED_EXTENSIONS` identifies a source file
by a real SHA-256 hash of its bytes (`identity: "sha256/1"` in the manifest
entry). Tier 3 is the exception: `_source_identity` never calls
`read_bytes()` on an audio/video source — it computes
`sha256("stat/1|{st_size}|{st_mtime_ns}")` from a single `os.stat()` call
instead, stored under the same `sha256` manifest key with a sibling
`identity: "stat/1"` field marking it as a stat fingerprint rather than a
content hash. This is deliberate: these files can be arbitrarily large and
are uncapped at discovery, so the engine has no business allocating a
multi-GB buffer just to hash bytes it then discards. `ensure_sidecar`,
`register_sidecar`, and `_entry_state` (backing `sidecar --list`) all route
through this one helper.

Trade-off: an in-place edit that happens to preserve both file size and
mtime produces a false cache hit under `"stat/1"` — impossible under
`"sha256/1"`, which would catch any content change. `retrieval extract
--force` (or `ensure_sidecar(..., force=True)`) is the escape hatch when
that matters.

## Section headings: `## Page N` and `## [HH:MM:SS] ...`

`_UNIT_HEADING_RE` (used for both unit counting and truncation boundaries)
matches two heading shapes: the PDF convention (`## Page N`) and the
time-coded convention used by caption/media transcripts (`## [HH:MM:SS]
label`). Both are treated identically by the chunker — heading density sets
chunk size — and both are honored by `register_sidecar`'s
oversized-transcript truncation (`_truncate_at_unit_boundary`, cutting only
at a whole-unit boundary, never mid-page or mid-section).

## Next steps

- [Customize indexing: PDF auto-indexing](../../how-to/customize-indexing.md#pdf-auto-indexing)
- [Persistence and cache: PDF sidecars](../persistence-and-cache.md)
- [Reference: project_loader API](project_loader.md)
