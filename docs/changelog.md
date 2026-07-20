# Changelog

## 0.2.0

- Persistent, on-disk index cache (`retrieval.persistence`) keyed by
  project path, with fingerprint-based staleness detection.
- New `retrieval` console-script CLI (`index` / `query` / `stats`),
  including `--force`, `--stale-ok`, and `--json`.
- `uv`-native setup — the engine's environment is managed entirely by
  `uv sync`/`uv run`; the deprecated no-`uv` fallback (`scripts/setup_venv.py`)
  has been removed.
- `all` extra — one-shot install of every optional retrieval extra
  (`local`, `remote`, `turbovec`, `pyserini`).

## 0.1.0

Initial vendored engine: `LexicalRetriever` (TF-IDF + BM25 + RRF),
`TurbovecRetriever` (dense ANN), `PiSeriniRetriever` (Lucene BM25), and the
heuristic/LLM contextualizers, with an in-memory-only engine API.

See [GitHub Releases](https://github.com/josix/agentic-retrieval/releases)
for the full release history and downloadable build artifacts.
