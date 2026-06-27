#!/usr/bin/env bash
#
# install.sh — set up the FDTDMEX `fdtdmex_mcp` MCP discovery server on macOS / Linux
# for standalone use with an MCP host (Claude Code, VS Code, Gemini/Antigravity, …).
#
# What it does (in order):
#   1. Verifies `uv` is available.
#   2. Installs the Python deps into .venv via `uv sync --extra "io,mcp"` (asks first).
#   3. Smoke-tests that the server imports and lists its 4 tools.
#   4. Writes a ready-made `.mcp.json` (the `fdtdmex` server block) next to this script.
#   5. Prints the registration commands for the common MCP hosts.
#
# Discovery only — it never runs a simulation and never touches the engine. Re-runnable.
# NOTE: inside the sibling ag-fdtd workspace you do NOT need this — ag-fdtd's UI launches the
# server for you. This installer is for using the server on its OWN, in any MCP client.

set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SERVER_DIR/.." && pwd)"   # pyproject lives at the repo root
ASSUME_YES=0
{ [ "${1:-}" = "-y" ] || [ "${1:-}" = "--yes" ]; } && ASSUME_YES=1

info() { printf '\033[1;34m[fdtdmex-mcp]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[fdtdmex-mcp]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fdtdmex-mcp]\033[0m %s\n' "$*" >&2; exit 1; }

confirm() {
  # confirm "question" -> 0 if yes. Auto-yes with -y or when non-interactive.
  if [ "$ASSUME_YES" = "1" ] || [ ! -t 0 ]; then return 0; fi
  printf '\033[1;36m[fdtdmex-mcp]\033[0m %s [Y/n] ' "$1"
  read -r reply
  case "$reply" in [nN]*) return 1;; *) return 0;; esac
}

# 1. uv (manages the server's Python environment)
if ! command -v uv >/dev/null 2>&1; then
  warn "uv is not installed. It manages this server's Python environment."
  cat <<'UVHELP'

  Install uv, then re-run ./install.sh. Official guide:
    https://docs.astral.sh/uv/getting-started/installation/

  macOS / Linux (standalone installer):
    curl -LsSf https://astral.sh/uv/install.sh | sh

  macOS (Homebrew):
    brew install uv

  (After installing, open a new shell or `source` your profile so `uv` is on PATH.)
UVHELP
  die "uv is required — install it (see above) and re-run."
fi
info "Using uv: $(command -v uv)"

# 2. dependencies (the io seam + the MCP server deps)
info "Dependencies (from pyproject 'io,mcp' extras): mcp, h5py, pydantic, plotly."
if confirm "Run 'uv sync --extra \"io,mcp\"' in $REPO_ROOT now?"; then
  ( cd "$REPO_ROOT" && uv sync --extra "io,mcp" )
  info "Dependencies installed."
else
  warn "Skipped sync. The server needs the 'mcp' (+ 'io') extras to run."
fi

# 3. smoke test — import + list the 4 contract tools
if ( cd "$REPO_ROOT" && uv run python - <<'PY'
import asyncio
from fdtdmex_mcp.server import mcp
names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
expected = {"list_solver_apis", "get_api_schema", "search_docs", "get_doc"}
assert expected <= set(names), f"missing tools: {expected - set(names)}"
print("tools:", names)
PY
) ; then
  info "Server import + 4-tool surface OK."
else
  warn "Smoke test failed — check that the sync above completed (needs the 'mcp' extra)."
fi

# 4. write a ready-made MCP config (the `fdtdmex` block) next to this script
MCP_JSON="$SERVER_DIR/.mcp.json"
cat > "$MCP_JSON" <<JSON
{
  "mcpServers": {
    "fdtdmex": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "$REPO_ROOT",
        "fdtdmex-mcp"
      ]
    }
  }
}
JSON
info "Wrote MCP config: $MCP_JSON"

# 5. next steps
cat <<NEXT

$(info "Setup complete. Register the server with your MCP host:")

  Claude Code CLI:
    claude mcp add fdtdmex -- uv run --directory "$REPO_ROOT" fdtdmex-mcp

  VS Code (native MCP / Copilot):
    Command Palette -> "MCP: Open User Configuration", then add the contents of:
    $MCP_JSON

  Claude VS Code extension / Cline / Roo / other:
    Paste the "fdtdmex" block from $MCP_JSON into the host's mcpServers config
    (e.g. ~/.claude.json for the Claude VS Code extension).

  Gemini CLI / Antigravity:
    gemini mcp add fdtdmex uv run --directory "$REPO_ROOT" fdtdmex-mcp

  Then restart the client and ask the agent to call list_solver_apis to confirm.
  Full tool reference + the corpus build: $SERVER_DIR/README.md
NEXT
