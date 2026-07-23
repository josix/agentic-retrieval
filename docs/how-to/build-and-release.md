# Build, distribute, and release

The engine (`engine/`) is a standard `hatchling`-backed Python package
(`agentic-retrieval`).

## Build a wheel + sdist

```bash
cd engine && uv build
```

This produces `engine/dist/agentic_retrieval-0.4.0-py3-none-any.whl` and the
matching `.tar.gz` sdist (gitignored — build artifacts, not checked in).

## Install the built wheel directly

```bash
uv pip install ./engine/dist/agentic_retrieval-0.4.0-py3-none-any.whl
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
