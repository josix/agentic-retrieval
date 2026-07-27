# Build, distribute, and release

The engine (`engine/`) is a standard `hatchling`-backed Python package
(`agentic-retrieval`).

## Build a wheel + sdist

```bash
cd engine && uv build
```

This produces `engine/dist/agentic_retrieval-<version>-py3-none-any.whl` and the
matching `.tar.gz` sdist (gitignored — build artifacts, not checked in).

## Install the built wheel directly

```bash
uv pip install ./engine/dist/agentic_retrieval-*.whl
```

The wheel registers a `retrieval` console script
(`[project.scripts]` in `engine/pyproject.toml`), so once the package is
published, `uvx` can run it without a local install or checkout:

```bash
uvx --from agentic-retrieval retrieval query "..." --root <path-to-project>
```

!!! note "PyPI gate"
    Not yet available — this only resolves once `agentic-retrieval` is
    published to PyPI; until then, use
    `uv run --project engine --extra all retrieval ...` above.

## Install from GitHub

Or install straight from GitHub, without building locally, using the
`engine/` subdirectory of this monorepo:

```bash
uv pip install "agentic-retrieval @ git+https://github.com/josix/agentic-retrieval.git#subdirectory=engine"
```

With optional extras:

```bash
uv pip install "agentic-retrieval[all] @ git+https://github.com/josix/agentic-retrieval.git#subdirectory=engine"
```

## Version bump checklist

The plugin and the engine are versioned in lockstep. Bump all four declaration
sites in a single commit — `engine/tests/test_version.py` fails CI if any
of them drift apart, or if `docs/changelog.md` has no section for the new
version.

1. `.claude-plugin/plugin.json` — `version`
2. `.claude-plugin/marketplace.json` — the `agentic-retrieval` entry's `version`
3. `engine/pyproject.toml` — `[project] version`
4. `engine/retrieval/__init__.py` — `__version__` (the only site with runtime
   consumers: it is written into each cache's `meta.json` as `engine_version`
   and shown by `retrieval stats` and `retrieval --version`)

Then:

5. Regenerate the lockfile: `uv lock --directory engine`. Never hand-edit
   `engine/uv.lock`.
6. Rename `docs/changelog.md`'s `## Unreleased` section to
   `## <version> — <YYYY-MM-DD>` and open a fresh empty `## Unreleased`
   above it.
7. Tag and push: `git tag v<version> && git push origin v<version>` — this
   is what triggers the release workflow below.

## Release workflow

**GitHub Release workflow**: tagged pushes (`v*`) trigger
`.github/workflows/release.yml`, which runs `uv build` in `engine/` and
attaches the resulting `dist/*.whl` + `dist/*.tar.gz` to the GitHub Release
— download those artifacts directly instead of building from source, if
you prefer. PyPI publishing is scaffolded in the same workflow but gated
off (`if: false`) until the package is ready to publish there.

## Manual docs deploy

The docs site (built with MkDocs Material) is deployed manually to GitHub
Pages — there is no automated deploy step yet:

```bash
uv run --project engine --group docs mkdocs gh-deploy -f mkdocs.yml
```

This builds the site and pushes it to the `gh-pages` branch of this
repository, publishing it at the `site_url` configured in `mkdocs.yml`.

## Next steps

- [Reference: CLI](../reference/cli.md)
- [Changelog](../changelog.md)
