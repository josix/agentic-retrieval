---
description: Compare contextual retrieval (plain and LLM-enriched), turbovec (dense ANN), pi-serini (Lucene BM25), hybrid (lexical + dense RRF), and tree-sitter (AST-boundary chunking) retrieval strategies
argument-hint: [setup [all|core|<extra>...] | index [--retriever all|<name>] | query "<text>"]
---

# Retrieval Command

Thin dispatcher — invoke the `retrieval` skill and follow its steps exactly.

## Subcommands

- `setup [all|core|<extra>...]` — `uv sync` the offline engine plus every
  optional per-strategy extra in one pass. `core` is not an installable
  extra — it means the base, no-extras sync (`uv sync` with no `--extra`),
  which is also the fallback tried first if the full sync fails
- `index` — build and persist an on-disk index for the invoking project
  (`retrieval index --root "$PROJECT_ROOT"`); defaults to building ALL
  strategy caches (lexical always succeeds; turbovec/pi-serini/hybrid/
  treesitter build too when their extras are present, else skipped with a
  note and exit 0) — each strategy keeps its own cache slot per project. Pass
  `--retriever lexical|turbovec|pi-serini|hybrid|treesitter` to build a
  single strategy instead (missing extras hard-fail with guidance, exit 1)
- `query "<text>"` — with no `--retriever` (the default, alias `--retriever
  all`), loads every available strategy's persisted index (auto-reindexing
  if missing or stale) and consolidates their rankings into a single
  deduplicated, explainable ranked list
  (`retrieval query "<text>" --root "$PROJECT_ROOT"`) — a handoff a
  following conversation/agent can act on directly. Pass
  `--retriever lexical|turbovec|pi-serini|hybrid|treesitter` to query one
  strategy instead (byte-identical output to before consolidation existed)
  — see the routing table below for when that's the right call

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

After setup, the skill searches the invoking project's own files with the
zero-dependency `LexicalRetriever`, via the `retrieval` console-script CLI
(see `skills/retrieval/SKILL.md` Step 2). For LLM or heuristic
contextualization before indexing, see the `remote` extra synced by setup,
plus the `lexical-retrieval-usage` / `hybrid-retrieval-usage` skills.

`setup` runs `uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all`
(see `skills/retrieval/SKILL.md` Step 1).

For per-method usage (setup, indexing/search snippets, graceful degradation,
and combining methods), see the sibling knowledge skills:
`lexical-retrieval-usage`, `dense-retrieval-usage`, `lucene-retrieval-usage`,
`code-retrieval-usage`, `hybrid-retrieval-usage`.

## Task

Invoke the `retrieval` skill and pass the arguments through unchanged: $ARGUMENTS
