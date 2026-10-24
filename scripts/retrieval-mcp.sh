#!/bin/sh
# Launch the agentic-retrieval MCP server (stdio) for the current project.
# stdout is reserved for the MCP protocol: nothing here may write to it.
PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "agentic-retrieval MCP: 'uv' not found on PATH (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

# Same default venv as /retrieval setup so installed extras are shared.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/agentic-retrieval/uv-venv}"

# `uv run` syncs inexactly: only adds the small `mcp` extra, keeps extras from setup.
exec uv run --project "$PLUGIN_ROOT/engine" --extra mcp retrieval mcp "$@"
