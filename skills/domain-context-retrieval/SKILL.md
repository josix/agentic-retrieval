---
name: domain-context-retrieval
description: This skill should be used when about to implement, modify, refactor or debug code in a project, or when asked "how is X done in this repo", "is there an existing pattern/convention for X", or "where is X implemented". Consults the agentic-retrieval MCP tools (search_domain_context, index_status, index_project) to ground the work in the repo's existing implementations before writing code, with a `retrieval query` CLI fallback.
---

# Domain Context Retrieval

Ground new work in what this repository already does. The plugin's MCP server
(`agentic-retrieval`, one per opened project) returns **evidence spans**
(`path:start_line-end_line` + snippet), not answers. Tool names below match
`engine/retrieval/mcp_server.py` exactly; in Claude Code they are prefixed
`mcp__plugin_agentic-retrieval_retrieval__`.

## When to use

- Before implementing, changing, refactoring or debugging code here.
- Before answering "how do we do X / what is the convention" from memory.
- Skip for trivial edits (typos, renames) where the target file is already known.

## Workflow

1. **`index_status`** (once per session). Read `recommendation` and `ready`.
   Nothing indexed is fine: lexical builds lazily on the first search.
2. **`search_domain_context(query)`** with a natural-language description of
   what you are about to build plus key identifiers (function names, domain
   nouns, error strings). Phrase it like the code/docs would, not like a chat
   request. Check `skipped` (strategies unavailable, each with a fix), `stale`
   / `index` (freshness) and `hint`.
3. **Read the hits.** Use `Read` on each cited `path:start_line-end_line`;
   snippets are leads (`truncated: true` means cut). Prefer hits with higher
   `agreement`/`confidence`; hits found by several retrievers are stronger.
4. **Reformulate once** (twice at most) if results are weak: swap in
   identifiers seen in the hits, synonyms, or a route per the table below.
   Raise `pool` (e.g. 50-100) and bias `weights` for recall-critical questions.
5. **Follow the existing pattern**: naming, error handling, tests. Cite the
   spans you relied on as `path:start-end`.
6. **No precedent?** Say so explicitly instead of inventing one.

## Retriever routing (same as `commands/retrieval.md`)

| Query characteristic | `retriever=` |
| --- | --- |
| Default; unsure | `all` (consolidated) |
| Exact keywords / identifiers / code tokens | `lexical` |
| Paraphrase / synonyms | `turbovec` |
| Lucene-grade BM25 depth | `pi-serini` |
| Vocabulary mismatch vs genuine irrelevance | `hybrid` |
| Code; AST spans + enclosing scope | `treesitter` |
| Chosen backend skipped/errored | `lexical` |

A single unbuilt heavy retriever returns empty `results` plus a `skipped`
reason, never an exception.

## When to call `index_project`

Only lexical works out of the box. If `skipped` lists strategies you need
(paraphrase-heavy questions, code-structure questions, a big repo), call
`index_project(retriever="all")` once (slow, one-off; add `auto=True` for a
corpus-calibrated baseline) or a single `turbovec` / `hybrid` / `treesitter`
build. After large edits, `index_project` refreshes stale heavy caches (the
plain lexical cache refreshes itself; a `lexical+ctx` index is never rebuilt
by MCP because that costs LLM tokens - ask the user, then run
`retrieval index --retriever lexical+ctx` via the CLI). Do not run it speculatively on huge repos.

## Fallback: MCP server unavailable

If the tools are missing or return `{"error": ...}` repeatedly, use the CLI as
in [`skills/retrieval/SKILL.md`](../retrieval/SKILL.md) (Step 2):

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
set -- --extra local --extra remote --extra turbovec --extra pyserini --extra treesitter --extra mcp
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}" \
uv run --project "${CLAUDE_PLUGIN_ROOT}/engine" "$@" \
  retrieval query "your query" --root "$PROJECT_ROOT" --auto
```

If that also fails, fall back to Grep/Read. See also the MCP section of
[`skills/retrieval/SKILL.md`](../retrieval/SKILL.md) and the `ground_implementation`
MCP prompt.
