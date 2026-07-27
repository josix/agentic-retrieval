# Quickstart

A five-minute tutorial: install `uv`, sync the engine, build an index over
your project, and run your first query.

## 1. Install uv

`uv` is a hard requirement — there is no fallback path.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm it's on `PATH`:

```bash
uv --version
```

## 2. Sync the engine

From a checkout of this repository:

```bash
uv sync --project engine --extra all
```

This one command syncs everything — the offline core plus every optional
extra (`remote`, `local`, `turbovec`, `pyserini`). Re-running it is safe:
`uv sync` is idempotent.

!!! note
    If the full sync fails on a heavy extra (most often `pyserini`, which
    needs a Java 21 JDK on `PATH`), sync core only to confirm the base
    engine works, then retry the full sync:

    ```bash
    uv sync --project engine              # core only — must always succeed
    uv sync --project engine --extra all  # retry full install
    ```

Expected output ends with a line similar to:

```
Resolved N packages in ...ms
Installed N packages in ...ms
```

## 3. Build an index

Capture your project root **before** invoking `uv run`, so you index your
own project's files, not the plugin's:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
uv run --project engine --extra all retrieval index --root "$PROJECT_ROOT"
```

Expected output:

```
indexed 42 docs -> /Users/you/project/.agentic-retrieval  fingerprint=<hash-prefix>
```

(The doc count and hash will differ for your project.) This persists a JSON
cache under `<project-root>/.agentic-retrieval`.

## 4. Query it

```bash
uv run --project engine --extra all retrieval query \
  "what carries data between networks" --root "$PROJECT_ROOT" --top-k 5
```

Expected output: one matching docid per line, best match first, e.g.:

```
README.md
docs/index.md
```

Re-running `query` loads the persisted index rather than rebuilding it,
unless your project's files changed since the last `index`/`query` call —
in which case it auto-reindexes first.

## Next steps

- [Install the plugin into Claude Code](../how-to/install.md) to drive the
  same flow with `/retrieval`.
- [CLI walkthrough](../how-to/index-and-query.md) — `--force`,
  `--stale-ok`, `--json`, `stats`, and the `/retrieval` subcommands.
- [Use each retriever](../how-to/use-each-retriever.md) — lexical,
  turbovec, and pi-serini in depth.
- [Full CLI reference](../reference/cli.md) for every flag.
