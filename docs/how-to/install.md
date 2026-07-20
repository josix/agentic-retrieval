# Install the plugin

How to install the `agentic-retrieval` plugin into Claude Code, plus a
brief note for other coding agents. This is a `uv`-only path — there is no
no-`uv` fallback.

## Prerequisites

- A current Claude Code build with plugin support. Check with
  `claude --version`; if the `/plugin` command isn't recognized once inside
  Claude Code, see [Troubleshooting](../reference/troubleshooting.md).
- **`uv`** — the hard requirement for setting up and invoking the engine.
  Check with `uv --version`; if missing, install it with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Optional: a Java 21 JDK on `PATH`, only if you want the `pyserini` extra
  (Pyserini/Anserini wraps a JVM) to actually work at index time.
- Network access, only for the optional `remote` / `local` / `turbovec` /
  `pyserini` extras — the core install is fully offline.

## Install path A (primary) — local marketplace

Use this when you have a local checkout of this repo (e.g. you cloned
`retrieval-skill` yourself, or a teammate handed you the folder).

1. Add the local folder as a marketplace, from inside Claude Code:

   ```
   /plugin marketplace add /path/to/retrieval-skill
   ```

   A relative path also works if your Claude Code session is already
   working near it:

   ```
   /plugin marketplace add ./retrieval-skill
   ```

2. Install the plugin from that marketplace:

   ```
   /plugin install agentic-retrieval@agentic-retrieval-marketplace
   ```

   (`agentic-retrieval-marketplace` is the marketplace's `name` field in
   `.claude-plugin/marketplace.json`; `agentic-retrieval` is the plugin's
   `name` field in `.claude-plugin/plugin.json` — both must match exactly
   for the install to resolve.)

!!! note
    Local/dev marketplaces have auto-update off, so pulling new commits into
    your checkout won't automatically update the installed plugin — re-run
    `/plugin marketplace add` on the same path (or reinstall) to pick up
    changes.

## Install path B — from GitHub (once published)

```
/plugin marketplace add josix/agentic-retrieval
```

This only resolves once the repo is published on GitHub under exactly that
name — `josix/agentic-retrieval` — matching the `repository` field in
`plugin.json` and the `source.repo` field in `marketplace.json`. The local
working copy on disk is named `retrieval-skill`; that folder name doesn't
need to match anything, but the *published GitHub repo* does need to be
named `agentic-retrieval` under the `josix` account for this exact
command to work.

## Scopes

`/plugin marketplace add` / `/plugin install` default to **user** scope
(available in every project you open in Claude Code). Install at **project**
or **local** scope instead if you only want the plugin active for one repo.
For a team, add the marketplace automatically for everyone via
`extraKnownMarketplaces` in a checked-in `.claude/settings.json`, so nobody
has to run the `/plugin marketplace add` step by hand.

## Activate and verify

After installing:

```
/reload-plugins
```

Then check it landed:

```
/plugin
```

Under **Installed**, `agentic-retrieval` should show 1 command and 5
skills.

Invoke it either namespaced...

```
/agentic-retrieval:retrieval
```

...or with the bare trigger the skill itself declares (`trigger: /retrieval`
in `skills/retrieval/SKILL.md`):

```
/retrieval
```

Both resolve to the same skill.

## First-run bootstrap

The engine's `uv`-managed environment doesn't exist until you sync it — run
this once:

```
/retrieval setup
```

This runs `uv sync --project "${CLAUDE_PLUGIN_ROOT}/engine" --extra all` —
one command syncs the offline core plus every optional extra. `uv sync` is
idempotent, so re-running `/retrieval setup` later is always safe.

!!! note
    If the full sync fails on a heavy extra (most often `pyserini`, which
    needs a Java 21 JDK on `PATH`), sync core only to confirm the base
    engine works, then re-run the full sync — see
    [index-and-query how-to](index-and-query.md) for the core-only
    fallback commands.

When installed as a plugin, Claude Code runs the plugin from
`~/.claude/plugins/cache`, not from your working copy on disk — so
`${CLAUDE_PLUGIN_ROOT}/engine` points inside that cache directory. `uv sync`
pins the resulting environment via `UV_PROJECT_ENVIRONMENT`, defaulting to
`$HOME/.cache/agentic-retrieval/uv-venv` — no action needed on your part
either way. See [environment variables](../reference/environment-variables.md)
for the full list.

## Uninstall / disable

```
/plugin uninstall agentic-retrieval@agentic-retrieval-marketplace
```

or, to keep it installed but turn it off:

```
/plugin disable agentic-retrieval@agentic-retrieval-marketplace
```

## Next steps

- [Integrate non-Claude coding agents](integrate-coding-agents.md)
- [Index and query with the CLI](index-and-query.md)
- [Troubleshooting](../reference/troubleshooting.md)
