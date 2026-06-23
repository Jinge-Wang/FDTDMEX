# server/ — `fdtdmex_mcp` MCP discovery server (WS-D)

A stdio [MCP](https://modelcontextprotocol.io) server that lets an agentic workspace (the sibling
**ag-fdtd** project) **discover** this solver's run API and a corpus of verified examples. It
speaks the same 4-tool contract as ag-fdtd's in-repo mock, so the agent can't tell them apart.

**Discovery only** — simulations run through `agentic_adapter/real_solver.py`, never through this
server (there is no blocking `sim_run` on the agent's path).

The four tools:

- `list_solver_apis(domain?)` — the run-API catalog (`run_fdtd_fdtdmex`).
- `get_api_schema(name)` — run-API params introspected **live** (the adapter's ring knobs + CLI
  contract + `fdtdmex.io.SceneModel`), so they can't drift from the code.
- `search_docs(query, limit?)` — BM25 search over a corpus generated from real `examples/`,
  `docs/`, the io schema, and docstrings → refs + snippets ([corpus.py](fdtdmex_mcp/corpus.py)).
- `get_doc(ref)` — the full, verbatim on-disk page.

## Installation (standalone — using it outside ag-fdtd, on macOS / Linux)

> Inside the sibling **ag-fdtd** workspace you do **not** need any of this — ag-fdtd's
> `./scripts/wire-fdtdmex.sh /path/to/FDTDMEX` points the agent at this server for you. The
> steps below are for using `fdtdmex_mcp` on its own, in any MCP client (Claude Code, VS
> Code, Gemini/Antigravity, …).

### 0. Install `uv`

This server's Python environment is managed by [`uv`](https://docs.astral.sh/uv/). If you
don't have it, follow the **[official installation guide](https://docs.astral.sh/uv/getting-started/installation/)**, e.g.:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
brew install uv                                   # macOS (Homebrew)
```

### 1. Run the installer

```bash
./server/install.sh        # verifies uv, syncs deps, smoke-tests, writes server/.mcp.json
```

It runs `uv sync --extra "io,mcp"` (after asking), confirms the 4-tool surface lists, writes
a ready-made `server/.mcp.json`, and prints the per-host registration commands. Pass `-y` to
skip the prompt. Equivalent manual steps:

```bash
uv sync --extra "io,mcp"
uv run fdtdmex-mcp                                # serve over stdio (or: python -m fdtdmex_mcp)
uv run python scripts/build_corpus.py --index    # force-rebuild the corpus + write llms.txt
```

### 2. Register with your MCP host

```bash
# Claude Code CLI
claude mcp add fdtdmex -- uv run --directory "$(pwd)" fdtdmex-mcp

# VS Code (native MCP / Copilot): Command Palette → "MCP: Open User Configuration",
# then paste the "fdtdmex" block from server/.mcp.json.
# Gemini CLI / Antigravity:
gemini mcp add fdtdmex uv run --directory "$(pwd)" fdtdmex-mcp
```

Restart the client and ask the agent to call `list_solver_apis` to confirm.

The corpus rebuilds lazily whenever a source file changes (cached under `~/.cache/fdtdmex_mcp/`).
See [../docs/mcp-and-ui.md](../docs/mcp-and-ui.md) and [../docs/agent-f-mcp-brief.md](../docs/agent-f-mcp-brief.md).
