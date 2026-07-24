# Consolidated query (the default)

`retrieval query "<text>"` with **no `--retriever` flag** runs every
available strategy in `_DEFAULT_INDEX_SET` (`lexical`, `turbovec`,
`pi-serini`, `hybrid`, `treesitter`), consolidates their per-retriever
rankings into a single deduplicated, ranked, and explainable list, and
prints (or writes) that list — a ranked set of entry points a following
conversation/agent explores and verifies from, without re-deriving
agreement/provenance itself. Any
strategy whose optional extras aren't installed is skipped, not a hard
failure; the run still exits `0` as long as `lexical` (the always-available
baseline) consolidates successfully.

```bash
uv run --project engine --extra all retrieval query \
  "what carries data between networks" --root "$PROJECT_ROOT" --top-k 5
```

`--retriever all` is an explicit alias for this same default. Pass
`--retriever <name>` (`lexical`, `lexical+ctx`, `turbovec`, `pi-serini`,
`hybrid`, `treesitter`) to query a single strategy instead — that path's
output is byte-identical to the pre-consolidation CLI (one
`path:start-end` per line, or the `{"query", "results": [...]}` JSON shape
with no `score`/`provenance` fields).

## How consolidation works

1. Each available strategy's cache is loaded (auto-reindexing if missing or
   stale, same as single-retriever `query`), and searched over a pool
   (`max(top_k * 3, 10)`) deeper than the requested `--top-k`.
2. `retrieval.consolidation.consolidate` groups hits whose `[start_line,
   end_line]` spans overlap or are line-adjacent within the same
   `source_path` (so, e.g., a `lexical` line-chunk and a `treesitter`
   AST-chunk over the same function merge into one candidate), then fuses
   each group's per-retriever ranks with a (optionally weighted) Reciprocal
   Rank Fusion.
3. The result is truncated to `--top-k` and printed/written as a single
   ranked list.

## Reading the output

Text mode (default) prints one line per result — the first token stays
`path:start-end` (so the `Read` affordance a downstream agent relies on
survives), followed by the fusion explanation:

```
retrieval/consolidation.py:120-160  [score=0.0328 agree=3/5 conf=high  via hybrid,lexical,treesitter]  Foo.bar
```

- `score` — the fused RRF score (higher is better).
- `agree=n/m` — `n` retrievers found this result, out of `m` retrievers
  actually consolidated this run (see `--json`'s `retrievers` list).
- `conf` — `high` (`agreement >= 2`), `medium` (a single dense/Lucene arm —
  `turbovec` or `pi-serini` — found it alone), or `low` (a single other arm
  found it alone).
- `via a,b,c` — the contributing retriever names (provenance), sorted.

Confidence and score reflect retriever agreement on your query text, not
whether the code is current or authoritative — verify a span against the
live file (and check for deprecation) before quoting it. Treat this list as
scaffolding for exploration, not a finished answer.

Skip notes (e.g. `turbovec: skipped (...)`) print to **stderr**, never
stdout, so stdout stays a clean ranked list either way.

`--json` emits an envelope built for a handoff:

```json
{
  "query": "what carries data between networks",
  "mode": "consolidated",
  "retrievers": ["lexical", "treesitter"],
  "skipped": [{"name": "turbovec", "reason": "turbovec retriever needs the 'turbovec' + 'local' extras: ..."}],
  "results": [
    {
      "docid": "a.py:1-10", "path": "a.py", "start_line": 1, "end_line": 10,
      "rank": 1, "context": "Foo.bar", "score": 0.0328,
      "provenance": ["lexical", "treesitter"], "agreement": 2, "confidence": "high",
      "contributors": [{"retriever": "lexical", "rank": 0, "docid": "a.py:1-10"},
                        {"retriever": "treesitter", "rank": 0, "docid": "a.py:2-8"}]
    }
  ]
}
```

Pass `--output PATH` to also persist that same envelope to disk — useful
when a following conversation/agent (or a script) should pick the ranking up
without re-running the query:

```bash
uv run --project engine --extra all retrieval query "..." --root "$PROJECT_ROOT" \
  --json --output /tmp/retrieval-handoff.json
```

## Weighting retrievers

Pass `--weights "name:weight,..."` to bias the fusion toward (or away from)
specific retrievers (unlisted retrievers default to weight `1.0`):

```bash
uv run --project engine --extra all retrieval query "..." --root "$PROJECT_ROOT" \
  --weights "turbovec:1.5,lexical:0.5"
```

See [Hybrid fusion](hybrid-fusion.md) for the underlying weighted-RRF math
(`retrieval.fusion.reciprocal_rank_fusion`).

## Agent-side follow-up (optional)

Consolidation is purely structural (span-merge + weighted RRF) — it never
makes an LLM call. Once a following agent has the consolidated list, it may
*optionally* apply its own listwise rerank over the top candidates (e.g. a
RankGPT-style "read the query plus the top-N snippets, ask the model to
re-order them") before acting — that is agent-side judgment on top of the
handoff, not an engine feature. See `skills/retrieval/SKILL.md` Step 3.

The recommended follow-up is the phased Q → R → T → C → S deep-answer
workflow documented in `skills/retrieval/SKILL.md` Step 3 (Decompose →
Retrieve per sub-question → Trace → Coverage gate → Synthesize): the
`--output` envelope above is exactly the per-sub-question seed carrier that
workflow's Phase R persists and Phase T traces from.

## Next steps

- [Hybrid fusion](hybrid-fusion.md)
- [Reference: consolidation API](../reference/api/consolidation.md)
- [Index and query with the CLI](index-and-query.md)
