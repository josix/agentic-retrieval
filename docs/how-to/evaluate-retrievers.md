# Evaluate retrievers

`retrieval eval` runs a labeled-query benchmark against every retriever
strategy plus the consolidated fusion, entirely in-memory (no
`<project-root>/.agentic-retrieval` writes) — the same graceful-degradation
convention as `retrieval index`/`query`: a strategy whose optional extras
are missing is skipped, not a hard failure, as long as the always-available
`lexical` strategy runs.

```bash
uv run --project engine retrieval eval --queries path/to/eval_queries.json
```

Use this to answer, with numbers instead of intuition, the question the
`retrieval` skill's routing table raises: *is a single `--retriever <name>`
query actually as good as the consolidated default, for the kinds of queries
you care about?* See the "Tradeoff" note in `skills/retrieval/SKILL.md` for
the qualitative version of this question.

## What it measures

For each labeled query, `retrieval eval`:

1. Builds every strategy (`lexical`, `turbovec`, `pi-serini`, `hybrid`,
   `treesitter`) over the query set's corpus, skipping any whose optional
   extras aren't installed.
2. Searches each surviving strategy, then consolidates their hits with
   `retrieval.consolidation.consolidate` (same as `query`'s default mode).
3. Scores **recall@k** and **nDCG@k** for every strategy and for
   `"consolidated"`, against the query's hand-labeled gold spans.
4. Profiles **cold** (first) and **warm** (median of repeated) search
   latency per strategy.
5. Tallies **confidence-signal validity**: across every query's consolidated
   top-k hits, what fraction of `high`/`medium`/`low` confidence hits are
   actually relevant (i.e. overlap a gold span) — this is the empirical
   check behind the consolidated default's `confidence` field.

## The labeled query set

A query set is a JSON file (see `engine/tests/fixtures/eval_queries.json`
for a working example):

```json
{
  "corpus_root": "eval_corpus",
  "queries": [
    {
      "id": "vm1",
      "category": "vocab-mismatch",
      "query": "how does a machine on the network get identified",
      "relevant": [
        {"source_path": "networks.txt", "start_line": 6, "end_line": 7}
      ]
    }
  ]
}
```

- `corpus_root` — a directory path, resolved **relative to the query set
  file's own directory** (override with `--root` to point at a different
  corpus without editing the file).
- `category` — free-form, but `vocab-mismatch` / `exact-keyword` / `code`
  are the conventional buckets used by the bundled fixture set (queries
  whose wording differs from the answer's wording, queries that reuse the
  answer's exact tokens, and code-shaped queries, respectively).
- `relevant` — one or more gold spans: `source_path` relative to
  `corpus_root`, plus the 1-based `[start_line, end_line]` range that
  answers the query. A hit counts as relevant if it shares the gold's
  `source_path` and its span overlaps (`hit.start <= gold.end and
  gold.start <= hit.end`); a hit with no line span (or a gold with none)
  falls back to matching on `source_path` alone.

## Reading the metrics

- **recall@k** — the fraction of a query's distinct gold spans overlapped by
  at least one of the strategy's top-k hits. `1.0` means every gold span was
  found somewhere in the top k.
- **nDCG@k** — binary-gain, rank-discounted: a hit earns full credit at rank
  1, less at each lower rank, and each gold span can only be credited once
  (so several redundant hits on the same span don't inflate the score).
  `1.0` is a perfect ranking (every gold span found, earliest possible).
- **confidence-signal validity** — precision (`relevant / total`) per
  `high`/`medium`/`low` bucket over every query's consolidated top-k hits.
  **Caveat**: with a small hand-labeled query set, these counts are small —
  treat single-digit-sample bucket precisions as directional, not
  statistically significant, and grow the query set before trusting them for
  a routing decision.
- **cold/warm latency** — cold is the first `search_detailed` call per
  query (includes any one-time warm-up cost); warm is the median of several
  repeated identical calls. **Caveat**: latency is machine-dependent —
  never compare absolute numbers across different runs/hardware, only
  relative cold-vs-warm or strategy-vs-strategy on the *same* machine in the
  *same* run.

## CLI reference

```bash
retrieval eval --queries PATH [--root PATH] [--k 5] [--warm-runs 5]
               [--json] [--output PATH]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--queries PATH` | required | Path to a labeled `eval_queries.json` file |
| `--root PATH` | the query set's own `corpus_root` | Override the corpus to eval against |
| `--k N` | `5` | recall@k / nDCG@k cutoff |
| `--warm-runs N` | `5` | Number of extra warm search repeats per query |
| `--json` | off | Emit a JSON report instead of text |
| `--output PATH` | none | Also write the JSON report to `PATH` |

Text mode prints a per-retriever aggregate table plus the
confidence-validity breakdown; `--json` emits
`{"k", "queries": [...], "aggregate": {...}, "confidence_validity": {...},
"skipped": [...], "build_s": {...}}` — see
`engine/retrieval/eval.py::report_to_dict` for the full per-query shape.

Exit code `0` as long as `lexical` (the always-available baseline) produced
results; skipped strategies (missing extras) are listed in `"skipped"`, not
a hard failure.

## Extending the query set

Add more queries (or a new corpus) to get a more reliable signal:

1. Add fixture files under a corpus directory.
2. For each new query, hand-pick the gold `source_path` + `[start_line,
   end_line]` span(s) that actually answer it — read the file yourself to
   get the line numbers right; a wrong gold span silently under- or
   over-counts every retriever's recall/nDCG.
3. Aim for a mix of `vocab-mismatch` (paraphrase, no shared vocabulary),
   `exact-keyword` (verbatim terms), and `code` (identifier/structure-shaped)
   queries — these stress the retrievers differently, which is the point of
   comparing them.
4. Re-run `retrieval eval` and compare the new aggregate against the
   previous one; a query set this small is not statistically rigorous, but
   it is enough to catch a strategy regressing or a routing assumption
   (e.g. "lexical is fine for code") being wrong.

## Next steps

- [Consolidated query (the default)](consolidated-query.md)
- [Hybrid fusion](hybrid-fusion.md)
- [Reference: CLI](../reference/cli.md)
