---
description: Compare contextual retrieval (plain and LLM-enriched), turbovec (dense ANN), pi-serini (Lucene BM25), hybrid (lexical + dense RRF), and tree-sitter (AST-boundary chunking) retrieval strategies
argument-hint: [setup [all|core|<extra>...] | index [--retriever all|<name>] | query "<text>"]
---

# Retrieval Command

Thin dispatcher — invoke the `retrieval` skill and follow its steps exactly.

## Subcommands

- `setup [all|core|<extra>...]` — `uv sync` the offline engine plus the
  per-strategy extras in one pass. The default sync is every strategy
  extra EXCEPT `pdf` (media extraction is delegated to the invoking
  agent, which authors PDF transcripts itself via `retrieval sidecar
  --register` — see `skills/retrieval/SKILL.md` Step 1); pass `pdf` (or
  `all`) explicitly to opt in to pypdf machine extraction instead.
  `core` is not an installable extra — it means the base, no-extras sync
  (`uv sync` with no `--extra`), which is also the fallback tried first
  if the full sync fails
- `index` — build and persist an on-disk index for the invoking project
  (`retrieval index --root "$PROJECT_ROOT" --auto`); defaults to building ALL
  strategy caches (lexical always succeeds; turbovec/pi-serini/hybrid/
  treesitter build too when their extras are present, else skipped with a
  note and exit 0) — each strategy keeps its own cache slot per project. Pass
  `--retriever lexical|turbovec|pi-serini|hybrid|treesitter` to build a
  single strategy instead (missing extras hard-fail with guidance, exit 1).
  Hyperparameters are the agent's call, made inline on this command:
  `--auto` derives a corpus-calibrated baseline, and explicit flags
  (`--tokenizer`, `--bm25-k1`, `--bm25-b`, `--code-chars`, `--bit-width`,
  `--embed-model`, `--lucene-k1`, `--lucene-b`, …) override per flag —
  precedence explicit > auto > static default. Decisions persist in each
  cache's meta and survive rebuilds (see `skills/retrieval/SKILL.md`
  Step 2, "Hyperparameters are YOUR decision")
- `query "<text>"` — with no `--retriever` (the default, alias `--retriever
  all`), loads every available strategy's persisted index (auto-reindexing
  if missing or stale) and consolidates their rankings into a single
  deduplicated, explainable ranked list
  (`retrieval query "<text>" --root "$PROJECT_ROOT"`) — a handoff a
  following conversation/agent can act on directly. Pass
  `--retriever lexical|turbovec|pi-serini|hybrid|treesitter` to query one
  strategy instead (byte-identical output to before consolidation existed)
  — see the routing table below for when that's the right call. Query-time
  knobs are also the agent's inline call: `--auto` resolves `top_k` from
  the index's persisted corpus stats (explicit `--top-k N` wins), and in
  consolidated mode `--weights "name:w,..."` biases retrievers matching
  the question's shape while `--pool N` deepens per-retriever candidate
  pools for recall-critical questions

## Agent routing — which retriever to query per question

| Query characteristic | Query with |
| --- | --- |
| Exact keywords / identifiers / code tokens; zero-dep default | `--retriever lexical` |
| Paraphrase / synonyms / wording differs from documents | `--retriever turbovec` |
| Lucene-grade BM25 depth/scale needed | `--retriever pi-serini` |
| Uncertain — vocabulary mismatch vs genuine irrelevance | `--retriever hybrid` |
| Code/script; want AST-boundary spans + enclosing scope | `--retriever treesitter` |
| Chosen backend skipped/errored | fall back to `--retriever lexical` |

See `hybrid-retrieval-usage` for the full decision walkthrough.

Don't leave PDF and other media (docx/pptx/xlsx/images) as placeholder
stubs — author their transcripts yourself and register with `retrieval
sidecar --register`; see `skills/retrieval/SKILL.md`'s "Author the media
transcript yourself (the default media path)" section for the full
workflow. This is the default path regardless of whether the `pdf` extra
is synced: a PDF without the `pdf` extra indexes as a stub until you
author its transcript (or you opt in to machine extraction instead), while
non-PDF media has no machine-extraction option at all — `sidecar
--register` is the only way to index it, `pdf` extra or not.

After setup, the skill searches the invoking project's own files with the
zero-dependency `LexicalRetriever`, via the `retrieval` console-script CLI
(see `skills/retrieval/SKILL.md` Step 2). For LLM or heuristic
contextualization before indexing, see the `remote` extra synced by setup,
plus the `lexical-retrieval-usage` / `hybrid-retrieval-usage` skills.

For explanatory/comprehensiveness questions, the skill's answer path runs the
phased deep-answer workflow (Q → R → T → C → S, Step 3 of
`skills/retrieval/SKILL.md`) rather than stopping at the first hit — this
command remains a thin dispatcher into that skill.

`setup` runs `uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra
local --extra remote --extra turbovec --extra pyserini --extra treesitter`
— deliberately without `--extra pdf`, so media extraction stays delegated
to the agent (see `skills/retrieval/SKILL.md` Step 1; add `pdf` to opt in
to pypdf).

For per-method usage (setup, indexing/search snippets, graceful degradation,
and combining methods), see the sibling knowledge skills:
`lexical-retrieval-usage`, `dense-retrieval-usage`, `lucene-retrieval-usage`,
`code-retrieval-usage`, `hybrid-retrieval-usage`.

## Task

Invoke the `retrieval` skill and pass the arguments through unchanged: $ARGUMENTS
