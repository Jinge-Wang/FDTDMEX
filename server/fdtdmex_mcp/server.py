"""FDTDMEX MCP discovery server — the real backing for the 4-tool discovery contract.

The agent's loaded tool surface stays SMALL and FIXED — FOUR discovery tools
(`list_solver_apis` / `get_api_schema` / `search_docs` / `get_doc`) — no matter how many
docs/examples exist. This is the real counterpart to ag-fdtd's in-repo mock
(`backend/app/mcp_server/__main__.py`): it speaks the IDENTICAL contract and the same
terse-text + trailing ``Next:`` convention, so the agent can't tell mock from real.

Discovery only. ag-fdtd runs simulations through its own adapter
(`agentic_adapter/real_solver.py`), NOT through this server — so there is no blocking
``sim_run`` on the agent's path here.

`get_api_schema("run_fdtd_fdtdmex")` is introspected **live** so it cannot drift from the
adapter: the authoring knobs come straight from ``real_solver._ring_knobs({})`` (the
adapter's own defaults), the run params mirror the adapter CLI contract, and the
sim_init payload fields come from ``fdtdmex.io.SceneModel``. `search_docs` / `get_doc`
serve a BM25 index over a corpus generated from real on-disk sources (see corpus.py).

`server.py` stays importable (no ``mcp.run`` here — that lives in ``__main__.py``).
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from . import corpus

mcp = FastMCP("fdtdmex-tools")


def _p(type_: str, desc: str, *, default=None, required=False) -> dict:
    return {"type": type_, "description": desc, "default": default, "required": required}


# --------------------------------------------------------------------------- #
# Solver-API manifest (discovery surface). One real run-API: run_fdtd_fdtdmex.
# --------------------------------------------------------------------------- #
_APIS: dict[str, dict] = {
    "run_fdtd_fdtdmex": {
        "name": "run_fdtd_fdtdmex",
        "domain": "fdtd",
        "solver": "fdtdmex",
        "summary": "FDTDMEX (Apple-MLX fork of fdtdx) — Metal forward FDTD. The agent authors an "
        "ag-fdtd SimConfig (ring knobs); the adapter translates it to an fdtdx Scene and "
        "runs sim_init → sim_run → S-matrix.",
        "returns": "Three contract files in --out-dir ({run-id}_result.hdf5, _summary.json, "
        "_preview.json) with a mode-resolved S-matrix in summary.scalars (per "
        "output_port·mode ← input_port·mode: mag, phase, transmission=|S|²); PROGRESS "
        "i/total streams on stdout.",
    },
}


# --------------------------------------------------------------------------- #
# Live introspection for get_api_schema (drift-proof against real_solver.py)
# --------------------------------------------------------------------------- #
def _ring_knob_defaults() -> dict:
    """The adapter's own default ring knobs — `real_solver._ring_knobs({})`, imported by
    path. Returns {} if the adapter can't be imported (degrade gracefully)."""
    adapter_dir = corpus.repo_root() / "agentic_adapter"
    if str(adapter_dir) not in sys.path:
        sys.path.insert(0, str(adapter_dir))
    try:
        import real_solver  # type: ignore

        return real_solver._ring_knobs({})
    except Exception:
        return {}


def _authoring_params() -> dict[str, dict]:
    """SimConfig knobs the agent authors, defaulted from the adapter's live values."""
    k = _ring_knob_defaults()
    gap = k.get("gap_um", 0.10)
    radius = k.get("R", 1.2)
    width = k.get("WG", 0.40)
    wl_um = round(k.get("wl", 1.55e-6) * 1e6, 6)
    res_nm = round(k.get("res", 20e-9) * 1e9, 6)
    return {
        "gap_um": _p("float", "Bus-ring coupling gap in µm (parameters.gap).", default=gap),
        "radius_um": _p("float", "Ring radius in µm (structures[Ring].radius).", default=radius),
        "width_um": _p("float", "Waveguide width in µm (structures[*].width).", default=width),
        "wavelength_um": _p("float", "Source center wavelength in µm (sources[*].wavelength).", default=wl_um),
        "grid_spacing_nm": _p(
            "float",
            "Uniform grid spacing in nm; floored at 20 nm for tractability (grid_spec.spacing).",
            default=res_nm,
        ),
    }


def _run_params() -> dict[str, dict]:
    """Adapter CLI run params (frozen CLI contract — real_solver.py main())."""
    return {
        "backend": _p(
            "str",
            "'mlx' (Apple-Metal forward engine) | 'mock' (GPU-free, schema-valid; the workspace default).",
            default="mlx",
        ),
        "steps": _p("int", "Spectrum sample count = wavelength points (adapter --steps).", default=11),
        "domain": _p("str", "PDE domain tag (adapter --domain).", default="fdtd"),
        "solver": _p("str", "Solver tag (adapter --solver).", default="fdtdmex"),
    }


def _scene_model_fields() -> list[str]:
    """The fdtdmex.io.SceneModel field names (sim_init payload), live. [] on failure."""
    try:
        from fdtdmex.io import SceneModel  # type: ignore

        return list(SceneModel.model_fields.keys())
    except Exception:
        return []


def _render_params(label: str, params: dict[str, dict]) -> list[str]:
    lines = [f"{label}:"]
    for pname, spec in params.items():
        req = " (required)" if spec.get("required") else f" (default: {spec.get('default')!r})"
        lines.append(f"  {pname}: {spec['type']}{req}")
        lines.append(f"      {spec['description']}")
    return lines


# --------------------------------------------------------------------------- #
# The 4 tools — terse text + trailing Next: hint (mirrors the ag-fdtd mock)
# --------------------------------------------------------------------------- #
@mcp.tool(
    description="List available PDE-solver run APIs (discovery surface). Optionally filter by "
    "domain, e.g. 'fdtd'. Returns names + one-line summaries; fetch one signature with "
    "get_api_schema, or find a verified setup with search_docs."
)
def list_solver_apis(domain: str | None = None) -> str:
    apis = [a for a in _APIS.values() if domain is None or a["domain"] == domain]
    lines = [f"solver run APIs{f' [domain={domain}]' if domain else ''}:", ""]
    for a in apis:
        lines.append(f"  {a['name']}  [{a['domain']}/{a['solver']}]")
        lines.append(f"      {a['summary']}")
    lines += [
        "",
        "Next: get_api_schema(<name>) for the signature, or search_docs(<query>) to find a verified setup / guide.",
    ]
    return "\n".join(lines)


@mcp.tool(
    description="Get the full parameter schema (types, defaults, required, returns) for ONE solver "
    "API by name. Use after list_solver_apis to write the run cell — don't guess params."
)
def get_api_schema(name: str) -> str:
    api = _APIS.get(name)
    if api is None:
        return f"unknown api {name!r}. Available: {', '.join(_APIS)} (see list_solver_apis)."
    lines = [f"# {api['name']}  [{api['domain']}/{api['solver']}]", api["summary"], ""]
    lines += _render_params("params (SimConfig knobs the agent authors)", _authoring_params())
    lines += [""]
    lines += _render_params("run params (adapter CLI contract)", _run_params())
    fields = _scene_model_fields()
    if fields:
        lines += ["", f"sim_init payload (fdtdmex.io.SceneModel fields): {', '.join(fields)}"]
    lines += [
        "",
        f"returns: {api['returns']}",
        "",
        "Next: build_simulation (sim_init) → commit_chapter → run_fdtd (sim_run) → collect_result → analyze.",
    ]
    return "\n".join(lines)


@mcp.tool(
    description="Search the documentation/example corpus (verified setups + guides) by free-text "
    "query. Returns ranked refs + one-line snippets — NOT full bodies. Fetch one with "
    "get_doc(ref). Prefer finding a verified example over authoring a config from scratch."
)
def search_docs(query: str, limit: int = 5) -> str:
    c = corpus.get_corpus()
    hits = c.search(query, limit=limit)
    if not hits:
        refs = ", ".join(c.refs())
        return f"no matches for {query!r}. Known refs: {refs}\nNext: get_doc(<ref>)."
    lines = [f"docs matching {query!r}:", ""]
    for h in hits:
        lines.append(f"  {h['ref']}  [{h['section']}]  {h['title']}")
        lines.append(f"      {h['snippet']}")
    lines += ["", "Next: get_doc(<ref>) for the full page (e.g. a copyable example setup)."]
    return "\n".join(lines)


@mcp.tool(
    description="Fetch ONE documentation/example page in full by its ref (from search_docs), e.g. "
    "'example/ring_mrm_oband'. Returns the real on-disk page text — a copyable example "
    "setup, or prose for a guide."
)
def get_doc(ref: str) -> str:
    c = corpus.get_corpus()
    doc = c.get(ref)
    if doc is None:
        refs = ", ".join(c.refs())
        return f"unknown ref {ref!r}. Known refs: {refs} (see search_docs)."
    lines = [f"# {doc['ref']} — {doc['title']}  [{doc['section']}]", "", doc["body"]]
    if doc["section"] == "examples":
        lines += ["", "Next: set the edited parameter with update_param, then build_simulation."]
    else:
        lines += ["", "Next: search_docs(<query>) for more, or list_solver_apis for the run API."]
    return "\n".join(lines)
