# Transcribe audio and video for indexing

Audio/video is the third media tier: no machine extractor exists for it
(unlike a PDF), and there's no native agent-read path either (unlike a
docx or image). Indexing it means running an external ASR tool and
registering the output as a sidecar transcript, the same mechanism as
[PDF auto-indexing](customize-indexing.md#pdf-auto-indexing) and
[Reference: CLI `sidecar`](../reference/cli.md#sidecar).

## Prerequisites

- **`ffmpeg`.** Extracts a mono 16 kHz WAV from a video container. macOS:
  `brew install ffmpeg`. Other platforms: your distro package manager.
- **whisper.cpp (`whisper-cli`).** The ASR engine. macOS:
  `brew install whisper-cpp`. Other platforms: build from the upstream
  whisper.cpp repository. Verify with `command -v whisper-cli`.
- **A `ggml` model file.** See [Download a model](#download-a-model).
- The plugin itself installed and synced — [Install the plugin](install.md).

!!! note
    whisper.cpp is one of three tools the skill accepts, in preference
    order: WhisperX → whisper.cpp → `whisper`. This page uses whisper.cpp
    because it needs no Python environment and writes SRT directly.
    Substituting another tool changes only the transcription step.

## Download a model

```bash
curl -L -o ggml-medium.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

Model size is a speed/accuracy trade-off, not a correctness one —
`small`/`medium` is usually enough for retrieval, since topical recall
matters more than exact wording. Reach for `large-v3` when the audio is
noisy or the content is jargon-heavy.

!!! note
    Keep the model file outside the project root you index. This is
    housekeeping, not a correctness requirement — a `.bin` file is not one
    of the discovered extensions, so it would never be indexed anyway.

## Find the files that need a transcript

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
uv run --project engine --extra all retrieval sidecar --list --root "$PROJECT_ROOT"
# media/recording.mp4: stub (agent-orchestrated) -> needs transcript
```

For a media-heavy corpus, filter with
`--json | jq '.files[] | select(.state != "stub")'` rather than scrolling
past a long wall of stubs in text mode.

## Extract 16 kHz mono audio

```bash
ffmpeg -i media/recording.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le recording.wav
```

- `-vn` drops the video stream.
- `-ac 1` downmixes to mono.
- `-ar 16000` resamples to the 16 kHz whisper.cpp expects.
- `-c:a pcm_s16le` writes uncompressed 16-bit PCM.

!!! warning
    Write the intermediate `.wav` and `.srt` OUTSIDE the indexed project
    root. `.wav` is itself a tier-3 suffix — leaving it inside the root
    would appear as a new agent-orchestrated stub — and `.srt` is
    tier-1 auto-extracted, so it would index independently and
    [double-count](#dont-double-count-a-captionvideo-pair) alongside the
    sidecar you register below.

## Transcribe

```bash
whisper-cli -m ggml-medium.bin -f recording.wav -l zh -osrt -of recording
```

`-l` is the source language of the audio (`zh` in the validated run above)
— set it to the recording's actual language. `-osrt -of recording` writes
`recording.srt`.

## Convert the SRT into the sidecar transcript grammar

The sidecar format is a specific grammar the chunker depends on, not free
prose:

- One `## [HH:MM:SS] Topic` heading roughly every 1-5 minutes of content
  — heading density sets chunk size, the same as `## Page N` for PDFs.
- A `[HH:MM:SS] ` prefix on **every** paragraph, using the absolute start
  time.
- Timestamps always zero-padded `HH:MM:SS` (`00:04:15`, not `4:15`).
- Never hand-write the `<!-- source: ... -->` header — `sidecar --register`
  adds it for you.

Illustrative example — a few SRT cues:

```srt
1
00:04:12,000 --> 00:04:15,500
Let's talk about how reciprocal rank

2
00:04:15,500 --> 00:04:18,200
fusion combines rankings from multiple

3
00:04:18,200 --> 00:04:21,000
retrievers into a single ordered list.
```

...become a transcript excerpt like:

```markdown
## [00:04:12] Fusion overview

[00:04:12] Let's talk about how reciprocal rank fusion combines rankings
from multiple retrievers into a single ordered list.
```

!!! warning
    Register only what the ASR tool actually output. You may reformat and
    re-segment freely into the grammar above, but never paraphrase,
    summarize, or fill in gaps the tool didn't transcribe. Mark a
    genuinely unintelligible stretch as `[inaudible]` rather than guessing.
    Never invent or promote a placeholder speaker label like `SPEAKER_00`
    into the transcript — diarization is optional, and only real,
    determinable speaker names belong in a transcript.

## Register and reindex

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
uv run --project engine --extra all retrieval sidecar --register media/recording.mp4 --transcript recording.md --root "$PROJECT_ROOT"
# reindex to pick up this sidecar: `retrieval index` (or just query)
```

A bare `retrieval query` auto-reindexes, since a registered transcript
changes the corpus fingerprint — `retrieval index` also works if you'd
rather reindex explicitly first.

## Verify

```bash
uv run --project engine --extra all retrieval sidecar --list --root "$PROJECT_ROOT"
# media/recording.mp4: agent-authored
```

`sidecar --list` now shows `agent-authored` instead of a stub. A query for
a distinctive phrase from the recording should hit
`.agentic-retrieval/extracted/media/recording.mp4.md` (the sidecar path,
not the original source):

```bash
uv run --project engine --extra all retrieval query \
  "reciprocal rank fusion" --root "$PROJECT_ROOT" --top-k 5
# .agentic-retrieval/extracted/media/recording.mp4.md:1-8  [score=0.0412 agree=5/5 conf=high  via hybrid,lexical,turbovec,pi-serini,treesitter]  Fusion overview
```

A strong match can show cross-retriever agreement like `agree=5/5`.

## Don't double-count a caption/video pair

`talk.mp4` (tier 3, stub until registered) and a sibling `talk.srt` (tier
1, auto-extracted) index as two independent sources — the engine
deliberately does not auto-link same-stem files. If a project ships both,
prefer the caption file and leave the video's stub unregistered rather
than authoring a redundant transcript for the mp4 too. There is no CLI
flag to exclude a file from discovery by pattern; that would be a
`discover_files(..., exclude_globs=...)` change at the call site, not a
`retrieval` command-line option today.

## If no ASR tool is available

!!! note
    Leave the stub as-is. This is an honest state, not an error.

## Next steps

- [Customize what gets indexed](customize-indexing.md#pdf-auto-indexing)
- [Reference: CLI `sidecar`](../reference/cli.md#sidecar)
- [Troubleshooting](../reference/troubleshooting.md)
