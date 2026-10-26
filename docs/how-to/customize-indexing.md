# Customize what gets indexed

`discover_files` / `load_documents` / `load_chunks`
(`retrieval/project_loader.py`) all accept the same keyword-only overrides:

| kwarg | Default | Purpose |
|---|---|---|
| `extensions` | `DEFAULT_EXTENSIONS` | Allowed file suffixes (e.g. `frozenset({'.md'})` to index only Markdown) |
| `exclude_dirs` | `DEFAULT_EXCLUDE_DIRS` | Directory names pruned during traversal (e.g. add `'docs-archive'`) |
| `exclude_globs` | `DEFAULT_EXCLUDE_GLOBS` | Filename deny-list globs (secret-looking names) |
| `include_basenames` | `DEFAULT_INCLUDE_BASENAMES` | Extensionless basenames allowed regardless of `extensions` (e.g. add `'justfile'`) |
| `max_bytes` | `MAX_FILE_BYTES` (1 MB) | Per-file size cap |
| `extract_max_bytes` | `extractors.EXTRACT_MAX_BYTES` (25 MB) | Per-file size cap for extractable suffixes (PDF, captions, agent-only media) — see [PDF auto-indexing](#pdf-auto-indexing) below. Agent-orchestrated audio/video is exempt from any size cap entirely. A source over this cap never appears in `retrieval sidecar --list` at all (excluded at discovery); raise this override for a large corpus |

## PDF auto-indexing

PDFs and caption files (`.srt`, `.vtt`) — plus a second tier of agent-only
media (`.docx`, `.pptx`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`)
and a third, agent-orchestrated tier of audio/video (`.mp4`, `.mov`, `.mkv`,
`.webm`, `.mp3`, `.m4a`, `.wav`, `.flac`) — under a project root are indexed
automatically: `extractors.EXTRACTABLE_EXTENSIONS` (the union of all three
tiers) is part of `DEFAULT_EXTENSIONS`, with no flag needed. Each such file
is routed through `retrieval.extractors` for a cached, sidecar-transcript
`Document` *before* the usual text-decode step, so it's chunked and
searched as prose rather than being silently skipped as binary.

- **Two size caps, plus an uncapped tier.** A PDF, caption file, or
  agent-only media file is capped at `extract_max_bytes` (25 MB) rather
  than the 1 MB `max_bytes` plain-text cap — these files are typically
  much larger than source files, and applying the small default cap would
  silently exclude every real-world one. Agent-orchestrated audio/video
  has **no size cap at all**: the engine never reads these files' bytes
  (see "Stat-only identity" below), so there's no memory-allocation reason
  to bound their size, and a multi-hour recording is exactly the case
  worth discovering.
- **Sidecar location and citation contract.** Each file's extracted text is
  written once to `<project-root>/.agentic-retrieval/extracted/<rel-path>.md`
  — always under the project root, deliberately ignoring
  `RETRIEVAL_INDEX_DIR`, since a sidecar is a citation target a coding agent
  `Read()`s by project-relative path. Search hits over such a file therefore
  cite the sidecar's path (e.g. `docs/paper.pdf` indexes as
  `.agentic-retrieval/extracted/docs/paper.pdf.md`), not the original file.
  A content-hash + extractor-version manifest alongside the sidecars makes
  re-extraction a no-op on unchanged files across `index --auto`'s multiple
  loader passes.
- **`--no-pdf`.** Pass `--no-pdf` to `retrieval index` to exclude PDFs
  *and* every other sidecar-routed media suffix from discovery entirely (an
  escape hatch for the auto-activated pipeline). The choice is persisted in
  the cache's meta and therefore sticky — a later flag-less `index`/`query`
  call keeps honoring it without needing the flag again.
- **Backend-missing stubs (PDF only).** Without the `pdf` extra (`pypdf`,
  included in the `all` extra) installed, every PDF still indexes — as a
  searchable placeholder stub naming the source file, instead of a real
  transcript — and a single `warning: pypdf is not installed` line prints
  to stderr per process. `index`/`query` never hard-fail on a missing
  `pypdf` backend; only the `retrieval extract` subcommand does (see
  [Reference: CLI](../reference/cli.md)).
- **Caption files (`.srt`/`.vtt`) — tier 1, no agent work needed.** A
  stdlib-only converter (no `pypdf`, no extra) merges cues into
  `[HH:MM:SS]`-prefixed paragraphs under `## [HH:MM:SS] <label>` section
  headings, indexed exactly like a PDF's `## Page N` sections. A caption
  file with zero cues or no recognizable `-->` cue-timing lines still
  indexes as a non-empty stub (`no-cues`/`malformed`).
- **Agent-only stubs (no machine extractor at all).** `.docx`, `.pptx`,
  `.xlsx`, and the image suffixes have no machine extractor — installing
  `pypdf` has no effect on them. Every such file always indexes as an
  `"agent-only"`-reason stub, and a single `warning: media files were
  indexed as agent-transcribable stubs` line prints to stderr per process.
  `retrieval sidecar --register` is the only way to index their real
  content.
- **Agent-orchestrated stubs (audio/video — no machine extractor, and no
  native agent-readable path either).** `.mp4`, `.mov`, `.mkv`, `.webm`,
  `.mp3`, `.m4a`, `.wav`, `.flac` always index as an
  `"agent-orchestrated"`-reason stub — an agent can't read these natively
  the way it can a docx or image, so it must orchestrate an external ASR
  tool (via Bash) and register the result. See the `retrieval` skill's
  "Audio and video: orchestrate an ASR tool, then register" section. For
  the end-to-end user workflow (ffmpeg, whisper.cpp, model download,
  register), see
  [Transcribe audio and video](transcribe-audio-video.md).
- **Stat-only identity (audio/video only).** The engine never reads a
  video/audio source's bytes for any purpose — `ensure_sidecar`,
  `register_sidecar`, and `sidecar --list` all compute its identity from a
  single `os.stat()` call instead (`sha256("stat/1|{size}|{mtime_ns}")`,
  manifest field `identity: "stat/1"`). Every other tier keeps a real
  content hash (`identity: "sha256/1"`). See [Persistence and cache: source
  identity](../reference/persistence-and-cache.md#source-identity-content-hash-vs-stat-only).
- **Agent-authored transcripts (no-install alternative for PDF; the only
  path for agent-only and agent-orchestrated media).** Instead of
  installing `pypdf` (or running ASR for audio/video), an agent can read
  the file natively (or run an ASR tool for audio/video) and hand-author
  its transcript, then register it directly: `retrieval sidecar --register
  <file> --transcript <file> --root <root>`. This writes the same kind of
  sidecar as `ensure_sidecar` would, stamped `extractor: agent-authored/1`
  instead of `pypdf-text/1`, and is a **durable** sidecar — it is not
  silently replaced if `pypdf` later gets installed (and never for
  agent-only/agent-orchestrated media, which has no machine-extraction
  equivalent to fall back to); only an explicit `retrieval extract --force`
  overwrites a PDF's (with a warning; `extract` never touches agent-only or
  agent-orchestrated media at all). It also mixes into the corpus
  fingerprint on its own (via the transcript's content hash), so
  re-registering a changed transcript over an otherwise-unchanged source
  file still triggers a reindex. See [Reference:
  CLI](../reference/cli.md#sidecar), [Transcribe audio and
  video](transcribe-audio-video.md), and the `retrieval` skill's "Author
  the media transcript yourself (the default media path)" and "Audio and video:
  orchestrate an ASR tool, then register" sections for the full workflows.
  For agent-only media specifically (`.docx`,
  `.pptx`, `.xlsx`, images), use this heading convention per format so the
  section breadcrumb is meaningful: `## Section: <heading>` (docx,
  fallback `## Part N`), `## Slide N — <title>` (pptx), `## Sheet: <name>`
  (xlsx), `## Image: <filename>` or `## Figure: <caption>` (images) — see
  `skills/retrieval/references/office-image-transcripts.md` for the full
  authoring guide.

## PO files (translation memory)

`.po` files are in `DEFAULT_EXTENSIONS` and are not size-chunked: the
chunk-granularity loaders (`load_chunk_documents`,
`load_ast_chunk_documents`, hence every CLI strategy) route them to
`retrieval.po_loader` (the built-in *record loader*, see
[below](#record-loaders-one-document-per-record)), which emits **one Document
per entry**. The span
(`path:start-end`) covers the whole entry block, so it feeds straight into
`Read`. The header entry and obsolete `#~` entries are skipped; fuzzy and
untranslated entries are indexed and flagged. `.pot` templates are not
discovered (they would duplicate every msgid as an untranslated entry).

Design decision: a PO Document's `text` is `msgid` + `msgstr` (plus
`msgctxt`/`msgid_plural`), so one entry matches both English and Chinese
queries (cross-lingual); `metadata` keeps them separate for display and
filtering.

| `metadata` key | Meaning |
|---|---|
| `kind` | always `"po_entry"` |
| `msgid`, `msgstr` | decoded strings (plural forms joined with newlines) |
| `fuzzy` | entry has the `fuzzy` flag |
| `translated` | `msgstr` is non-empty (all plural forms for plural entries) |
| `line` | line of the `msgid` keyword |
| `references` | `#:` source references |
| `msgctxt`, `msgid_plural`, `msgstr_plural` | present only when the entry has them |

Chinese text needs the `cjk` tokenizer (the default `\w+` keeps a whole
Chinese sentence as one token). Build and query with it, then filter
fuzzy/untranslated hits from the JSON `metadata`:

```bash
retrieval index --root <po-repo> --retriever lexical --tokenizer cjk
retrieval query "疊代器" --root <po-repo> --retriever lexical --top-k 10 --json \
  | jq '.results[] | select(.metadata.translated and (.metadata.fuzzy|not))
        | {path, start_line, msgid: .metadata.msgid, msgstr: .metadata.msgstr}'
```

Use `--retriever lexical` on both commands: a consolidated `query` after a
lexical-only index builds any missing strategy with the `plain` tokenizer.
Untranslated entries contain only English text and are short, so they can
rank high for English queries; raise `--top-k` and filter on
`metadata.translated`. Single-character Chinese queries cannot match inside a
longer run (bigram indexing). The legacy `load_chunks` path still chunks
`.po` files by size. Re-index existing projects (and any cache built with
`--no-pdf` before 0.11.0) to pick up `.po` files.

## Record loaders: one Document per record

Any record-oriented format (CSV glossary, JSONL, SRT-style cues) can be
indexed per record without touching the engine core. Write a parser module
and register a suffix:

1. Write a module whose `load(docid, text, url)` returns one `Document` per
   record. Each needs `docid="{path}:{start}-{end}"`, `source_path`, the
   line span, and `metadata["kind"]`.
2. At the end of the module call
   `record_loaders.register(RecordLoader(kind=..., suffixes=frozenset({".csv"}), load=..., detail_keys=(...)))`.
   `detail_keys` are the `metadata` keys printed as indented `label: value`
   lines in CLI text mode; pass `describe=` only for custom formatting, as
   `retrieval.po_loader` does.
3. For an in-tree module, add it to `record_loaders.BUILTIN_LOADER_MODULES`.
   For an out-of-tree module, import it and pass
   `extensions=DEFAULT_EXTENSIONS | {".csv"}`: `DEFAULT_EXTENSIONS` is
   computed at first import of `project_loader`, so later registrations are
   routed but not discovered by default.

`retrieval.po_loader` is the worked example. Translation-specific policy
(filtering fuzzy entries, assembling few-shot examples) belongs to callers,
not the engine.

```bash
PROJECT_ROOT=<project> python - <<'PY'
import os
from retrieval import record_loaders
from retrieval.project_loader import DEFAULT_EXTENSIONS, load_chunk_documents
import my_csv_loader  # calls record_loaders.register(...) at import

docs = load_chunk_documents(os.environ["PROJECT_ROOT"], extensions=DEFAULT_EXTENSIONS | {".csv"})
print(len(docs), sorted(record_loaders.registered_suffixes()))
PY
```

## Tune hyperparameters

By default every strategy uses fixed hyperparameters (chunk sizes, BM25
`k1`/`b`, embedding model, and so on). `retrieval tune` derives better
values from your corpus, and `index --auto` applies them.

```bash
# 1. Preview: writes no index cache (may write extracted-transcript sidecars and their manifest)
uv run --project engine --extra all retrieval tune --root "$PROJECT_ROOT"

# 2. Build with the derived values
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT" --auto

# 3. See what was recorded (hyperparams and corpus_stats lines)
uv run --project engine --extra all retrieval stats --root "$PROJECT_ROOT"

# Override one value; everything else stays auto-derived
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT" --auto --code-chars 800
```

`tune` measures two things: filesystem signals (file extensions, the share of
bytes in tree-sitter-mapped code files as `code_fraction`, the CJK share of
word characters as `cjk_fraction`, median code-file length) and chunk-token
statistics after chunking (mean, coefficient of variation, p10/p50/p90). The
rules:

| Parameter | Rule | Static default |
| --- | --- | --- |
| `tokenizer` | `cjk` if `cjk_fraction` >= 0.1 (wins over code); else `code` if `code_fraction` > 0.3; else `plain` | `plain` |
| `prose_chars` / `default_chars` | `400` | `400` |
| `config_chars` | `700` | `400` |
| `code_chars` | `1200` if `code_fraction` > 0.3, else `400` | `400` |
| `ast_max_chars` (`treesitter` only) | `2000` if the median code file exceeds 400 lines, else `1200` | `1200` |
| `bm25_k1` | `1.2` if `code_fraction` > 0.6; `1.5` if >= 0.2; `1.8` if median chunk tokens > 150; else `1.5` | `1.5` |
| `bm25_b` | `1.0` if chunk-size CV > 1.2 or p90/p50 > 4; `0.75` if CV >= 0.4; else `0.3`; `0.75` without chunk stats | `0.75` |
| `model_name` | `flax-sentence-embeddings/st-codesearch-distilroberta-base` if `code_fraction` > 0.6; `sentence-transformers/all-mpnet-base-v2` if <= 2000 chunks; else `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` |
| `bit_width` | `4`, `3` or `2` as chunks x embedding dimensions is <= 5e7, <= 5e8 or larger | `4` |
| `lucene_k1` / `lucene_b` | `0.9` / `0.4` | `0.9` / `0.4` |

Things to know:

- **Precedence**: an explicit flag beats the `--auto` value, which beats the
  static default.
- **Flags without `--auto`** are layered over the heuristics' empty-corpus
  fallbacks, which match the static defaults except `config_chars` = `700`.
- **Sticky**: recorded `hyperparams` and `corpus_stats` are reused by later
  rebuilds that pass no `--auto` or flags (including a `query`-triggered
  reindex).
- **Re-tuning rebuilds**: passing `--auto` or flags whose resolved values
  differ from a cache's recorded ones makes that cache stale, so it is rebuilt
  without `--force`.
- **Thin chunks**: `tune` warns (advisory only) when the p10 chunk size is
  below 15 tokens.
- **`query --auto`** only picks `--top-k` from the recorded `corpus_stats`; it
  never re-derives index-time values.
- **`eval --auto`** derives parameters from the eval corpus and records them
  as `auto_params` (see [Evaluate retrievers](evaluate-retrievers.md)).
- **MCP**: `index_project(auto=True)` exposes `--auto` only, not the
  per-hyperparameter flags.

## Extend, don't replace

Extend a `DEFAULT_*` constant with `|` rather than replacing it, so you keep
the built-in exclusions:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJECT_ROOT="$PROJECT_ROOT" uv run --project engine --extra all python - <<'PY'
import os

from retrieval.project_loader import DEFAULT_EXCLUDE_DIRS, load_documents

root = os.environ["PROJECT_ROOT"]
no_archive = load_documents(root, exclude_dirs=DEFAULT_EXCLUDE_DIRS | {"docs-archive"})
print("excluding docs-archive:", len(no_archive))
PY
```

Replacing `DEFAULT_EXCLUDE_DIRS` outright (instead of unioning with `|`)
silently drops the built-in exclusions — `.git`, `.venv`, `node_modules`,
etc. would then be walked and indexed.

## Default exclusions summary

- VCS/venv/build/cache directories (`.git`, `.venv`, `node_modules`,
  `__pycache__`, `dist`, `build`, `.ruff_cache`, etc.)
- Filenames that look like secrets by name (`.env`, `*.pem`, `*.key`,
  `id_rsa*`, `*credentials*`, `*secret*`, etc. — filename matching only, not
  content scanning)
- Binary/undecodable files
- Anything over 1 MB — except an extractable suffix (PDF, captions,
  agent-only media), capped at 25 MB instead, or agent-orchestrated
  audio/video, which is uncapped entirely (see [PDF
  auto-indexing](#pdf-auto-indexing) above)

!!! warning
    The filename deny-list (`DEFAULT_EXCLUDE_GLOBS`) is a **best-effort**
    guard against accidentally indexing credential files by name — it is
    not content scanning. A file named `notes.txt` containing an API key
    will still be indexed; do not rely on this module for secret detection.

Extensionless well-known project files (`Dockerfile`, `Makefile`, `LICENSE`,
`README`, `CHANGELOG`, etc.) are included via a basename allowlist, since the
extension allowlist alone would otherwise skip them.

Full kwargs reference and more examples:
`skills/lexical-retrieval-usage/SKILL.md` ("Customizing file discovery").

## Next steps

- [Reference: project_loader API](../reference/api/project_loader.md)
- [CLI reference: tune](../reference/cli.md#tune)
- [Architecture](../concepts/architecture.md)
