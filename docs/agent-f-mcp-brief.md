# Agent F brief — build the FDTDMEX MCP discovery server

**Self-contained.** You are the FDTDMEX-repo agent. Build the real MCP **discovery** server
for this repo so the sibling **ag-fdtd** agentic workspace can discover this solver's API and
docs over stdio. You do **not** need ag-fdtd's source — only the frozen contract below. The
companion ag-fdtd agent develops against an in-repo mock that implements the *same* contract,
so you two never block each other; you meet at integration (a wire script + a live demo).

> This brief is derived from the cross-repo plan
> `~/.claude/plans/ag-fdtd-dev-docs-final-lap-plan-md-the-gleaming-aurora.md`. The two
> "Execution brief" halves are independent; this is the FDTDMEX half.

---

## Hard rules

- **Do NOT touch the simulation engine** (`src/fdtdx`, the io seam internals). The MCP server,
  a translation layer, a docs corpus, packaging, and docs are all fair game.
- **Discovery only.** ag-fdtd runs simulations through a separate adapter
  (`agentic_adapter/real_solver.py`, already built), NOT through this MCP server. Do **not**
  expose a blocking `sim_run` as the agent's path. (You MAY add `sim_init`/`sim_run`/
  `sim_postproc` tools for other clients, but they are not part of the contract below and
  ag-fdtd ignores them.)
- **Speak the universal vocabulary exactly** (names/signatures/return style below). The
  vocabulary is ag-fdtd's and is canonical — do not rename it to something fdtdx-native. If a
  name/signature/return shape genuinely must change, it changes in the shared plan's contract
  **first**, then both repos update.
- **Tiny loaded surface.** Exactly the 4 tools — terse one-line descriptions, detail fetched
  on call. (Reference `~/Projects/MESA/mesa-mcp` for the retrieval pattern, but it has far more
  tools than we want — do not copy its breadth.)
- **No hardcoded examples.** `search_docs`/`get_doc` serve a corpus **generated from real
  source** (this repo's `examples/`, docstrings, the io schema). Never retype an example as a
  string literal.

---

## THE FROZEN CONTRACT — 4 universal tools

Implement these with the `mcp` package's `FastMCP`. Return **terse formatted text** (not raw
JSON) with a trailing `Next:` hint. Full bodies are fetched only on call (token budget is
first-class).

| tool | signature | returns | backed by |
|------|-----------|---------|-----------|
| `list_solver_apis(domain: str \| None = None)` | optional `"fdtd"` filter | one line per run-API (name + one-line summary) + `Next:` hint | a solver capability manifest |
| `get_api_schema(name: str)` | e.g. `"run_fdtd_fdtdmex"` | full run-API param schema: names, types, defaults, required, returns | **live** introspection — the params the adapter accepts + the `fdtdmex.io` schema; must not drift from `real_solver.py` |
| `search_docs(query: str, limit: int = 5)` | NL query or term | ranked **refs + one-line snippets** (ref, section, title) — NOT full bodies | a **BM25 index over the generated corpus** |
| `get_doc(ref: str)` | a ref returned by `search_docs` | the full page text — a copyable setup for examples, prose for guides | the corpus / real `examples/` files |

Expected agent flow: `list_solver_apis("fdtd")` → `get_api_schema("run_fdtd_fdtdmex")` →
`search_docs("ring resonator")` → `get_doc("example/ring_mrm_oband")` → (the agent then authors
an ag-fdtd `SimConfig` and runs via the adapter — not your concern).

**Reference return style** (match this shape): see ag-fdtd's in-repo mock, which implements the
identical 4 tools, at `~/Projects/Kronos/ag-fdtd/backend/app/mcp_server/__main__.py`. Mirror its
terse-text + `Next:` convention so both servers feel identical to the agent.

### `get_api_schema` — the run-API params (what the agent actually authors)

The agent authors an ag-fdtd `SimConfig` (ring knobs etc.); the adapter translates
`SimConfig → fdtdx Scene`. So `get_api_schema("run_fdtd_fdtdmex")` must reflect **the params the
adapter accepts**, derived live from:
- `agentic_adapter/real_solver.py` — the accepted knobs (`_ring_knobs`: gap, radius, width,
  wavelength, grid spacing) + run params (`backend` ∈ {mlx, mock}, steps/wavelength/resolution).
- `src/fdtdmex/io` — the io seam schema for anything that flows into `sim_init`.

Do not introspect deep `fdtdx` scene-object types **for the agent's authoring surface** — those
feed the corpus, not `get_api_schema`. (FYI: scene objects are `pytreeclass`, introspectable via
`pytreeclass.fields()`; the authoritative accepted-type whitelist is the local
`valid_object_names`/`valid_constraint_names` set inside `JsonSetup.validate()` in
`src/fdtdx/conversion/json.py`. Use that only if you choose to enrich the corpus.)

---

## The documentation/example corpus ("agents create all documentation")

A regenerable corpus + BM25 index that `search_docs`/`get_doc` serve.

- **Source of truth (real, not retyped):** `examples/ring_mrm_oband/` (the verified O-band ring
  study — both `ring_mrm_oband.py` and its README), fdtdx/fdtdmex docstrings, the `fdtdmex.io`
  `SceneModel`/`JsonSetup` schema, the adapter's accepted knobs, and relevant `docs/*.md`.
  Surface examples like mesa's `test_suite` cases — discovered from disk, returned as the **real
  file content**, with a stable `ref` (e.g. `example/ring_mrm_oband`, `guide/coupling_gap`).
- **Build step (regenerable):** `scripts/build_corpus.py` ingests the above → a markdown/JSON
  corpus + an `llms.txt`-style curated index → a **BM25 index**. Mirror the proven approach in
  `~/Projects/MESA/mesa-mcp/mesa_mcp/docs/index.py` (walk + chunk; cache as JSON keyed by a
  cheap content signature: file count / total size / max mtime; build lazily on first search).
  Document how to regenerate it.
- **Location:** keep it in this repo (`server/fdtdmex_mcp/corpus/` or a user cache dir). It must
  not require importing ag-fdtd. Optionally also expose the curated index as an **MCP resource**
  for cheap bulk loading.

---

## Build tasks

1. **Server** next to the existing stub `server/fdtdmex_mcp/__init__.py` (keep the docstring):
   - `server/fdtdmex_mcp/server.py` — `mcp = FastMCP("fdtdmex-tools")` + the 4 `@mcp.tool`
     functions, terse text + `Next:` hints.
   - `server/fdtdmex_mcp/__main__.py` — `def main(): mcp.run(transport="stdio")` and the
     `if __name__ == "__main__": main()` guard. Keep `__main__` trivial so `server.py` is
     importable in tests.
   - `server/fdtdmex_mcp/corpus.py` (+ a `docs/` helper or reuse) — corpus loader + BM25 search.
2. **Doc-gen** — `scripts/build_corpus.py` (regenerable), per the corpus section.
3. **Packaging** in `pyproject.toml` — ship `server/` as a second wheel package + a console
   script (the cleanest install; `import fdtdmex_mcp` then works without a path hack):
   ```toml
   [project.scripts]
   fdtdmex-mcp = "fdtdmex_mcp.__main__:main"

   [tool.hatch.build.targets.wheel]
   packages = ["src/fdtdx", "src/fdtdmex", "server/fdtdmex_mcp"]

   [tool.hatch.build.targets.wheel.sources]
   "server/fdtdmex_mcp" = "fdtdmex_mcp"
   ```
   Also drop `server` from the ruff `exclude` now that it ships code. The `[mcp]` extra already
   pins `mcp>=1.0` + h5py/pydantic/plotly — no dependency change needed.
4. **First step before coding:** `uv sync --extra mcp` — the `mcp` package is declared in
   `[mcp]` but is **not yet installed** in `.venv`; `from mcp.server.fastmcp import FastMCP`
   fails until you sync.
5. **(optional)** add a `fdtdmex-solver` console script for `agentic_adapter/real_solver.py` so
   it's invokable by name as well as by path. The adapter is **already implemented and
   validated** against `--backend mock` — do **not** rewrite it; only confirm its CLI still
   matches the contract below.
6. **README** — document `uv sync --extra "io,mcp"`, `uv run fdtdmex-mcp`, and the corpus build.

---

## Adapter CLI contract (confirm only — do NOT change the adapter)

ag-fdtd's `LocalJobRunner` spawns the adapter with exactly:

```
--bundle <path> --out-dir <dir> --run-id <id> --steps <int> --domain <str> --solver <str> --backend <mlx|mock> [--fail-mode <diverge|mesh_fail>]
```

and expects three files written to `--out-dir`:
`{run-id}_result.hdf5`, `{run-id}_summary.json`, `{run-id}_preview.json`, with `PROGRESS i/total`
lines streamed to stdout. `real_solver.py` already does this; just confirm it still matches.

---

## Standalone done-check (verify ALONE, no ag-fdtd needed)

1. `uv run fdtdmex-mcp` starts and serves over stdio.
2. A stdio MCP client lists **exactly** `['list_solver_apis','get_api_schema','search_docs','get_doc']`.
3. `list_solver_apis("fdtd")` includes `run_fdtd_fdtdmex`; `get_api_schema("run_fdtd_fdtdmex")`
   shows the adapter-accepted params (no guessed fields).
4. `search_docs("ring resonator")` returns a ref like `example/ring_mrm_oband`; `get_doc(<ref>)`
   returns the **real** example text (assert the returned content matches the on-disk
   `examples/ring_mrm_oband/` file — proving it's generated, not hardcoded).

Example stdio smoke (after `uv sync --extra mcp` + an editable/wheel install):

```bash
uv run --extra mcp python - <<'PY'
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client(StdioServerParameters(command="fdtdmex-mcp")) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            print(await s.call_tool("search_docs", {"query": "ring resonator"}))
asyncio.run(main())
PY
```

---

## Files you'll touch

- `server/fdtdmex_mcp/{__init__.py (keep), server.py (new), __main__.py (new), corpus.py (new)}`
- `scripts/build_corpus.py` (new)
- `pyproject.toml` (`[project.scripts]` + wheel `packages`/`sources`; ruff `exclude`)
- `README.md` (install + MCP + corpus)
- (read-only references) `examples/ring_mrm_oband/`, `agentic_adapter/real_solver.py`,
  `src/fdtdmex/io/`, `docs/mcp-and-ui.md`; and the ag-fdtd mock at
  `~/Projects/Kronos/ag-fdtd/backend/app/mcp_server/__main__.py` for the return-style reference.

## Coordination

The ag-fdtd agent is building the matching wiring + mock-parity against the same contract. The
**only** shared artifact is the 4-tool contract above. Integration is a one-shot
`wire-fdtdmex.sh` (on the ag-fdtd side) + a live demo. If you need to change the contract,
change it in the shared plan first and flag the ag-fdtd side.
