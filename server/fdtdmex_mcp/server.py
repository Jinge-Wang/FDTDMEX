"""FDTDMEX MCP discovery server — the real backing for the 4-tool discovery contract.

The agent's loaded tool surface stays SMALL and FIXED — FOUR discovery tools
(`list_solver_apis` / `get_api_schema` / `search_docs` / `get_doc`) — no matter how many
docs/examples exist. This is the real counterpart to ag-fdtd's in-repo mock
(`backend/app/mcp_server/__main__.py`): it speaks the IDENTICAL contract and the same
terse-text + trailing ``Next:`` convention, so the agent can't tell mock from real.

Discovery only. ag-fdtd executes **natively** in its own notebook kernel (running in this
repo's venv): the agent writes native fdtdmex Python (`Scene` / `pack` / `run_simulation_from_hdf5` /
`sim_postproc` / `compute_mode`) and runs it there — this server teaches *what to write*, it
never runs a simulation. The launch is **non-blocking**: `run_simulation_from_hdf5` stages a job
folder and detaches the solver, so there is no blocking run on the agent's path here.

`get_api_schema(name)` describes the **native** pack/launch/post-proc entry points and is introspected
**live** so it cannot drift: each signature comes straight from ``inspect.signature`` of the real
function (``fdtdmex.io.{pack,run_simulation_from_hdf5,sim_postproc}`` /
``fdtdx.core.physics.modes.compute_mode``) and the pack payload fields come from
``fdtdmex.io.SceneModel``. `search_docs` / `get_doc` serve a BM25 index over a corpus generated from
real on-disk sources (see corpus.py).

`server.py` stays importable (no ``mcp.run`` here — that lives in ``__main__.py``).
"""

from __future__ import annotations

import importlib
import inspect

from mcp.server.fastmcp import FastMCP

from . import corpus

mcp = FastMCP("fdtdmex-tools")


# --------------------------------------------------------------------------- #
# Solver-API manifest (discovery surface). The NATIVE authoring API the agent
# writes in its own kernel: pack → run_simulation_from_hdf5 → sim_postproc, plus
# compute_mode. Each entry's signature is introspected live (drift-proof) in
# get_api_schema. The blocking primitive run_simulation is deliberately NOT here.
# --------------------------------------------------------------------------- #
_APIS: dict[str, dict] = {
    "pack": {
        "name": "pack",
        "domain": "fdtd",
        "solver": "fdtdmex",
        "import": ("fdtdmex.io", "pack"),
        "summary": "Resolve a declarative fdtdx Scene / SceneModel and pack it into a project folder "
        "(location) as one self-contained config HDF5 (content-addressed) plus a lightweight editable "
        "config JSON. Heavy lifting (place objects, rasterize, freeze sources/detectors) happens here. "
        "One packed bundle can back many runs.",
        "returns": "PackResult(hdf5_path, config_path, config_hash); os.fspath(result) is the HDF5 path "
        "you hand to run_simulation_from_hdf5.",
    },
    "run_simulation_from_hdf5": {
        "name": "run_simulation_from_hdf5",
        "domain": "fdtd",
        "solver": "fdtdmex",
        "import": ("fdtdmex.io", "run_simulation_from_hdf5"),
        "summary": "Launch a packed config HDF5 as a job. NON-BLOCKING: stages a job folder under "
        "parent_folder (copies the bundle, snapshots config.json, makes outputs/), writes status.json, "
        "and launches the solver DETACHED — returns immediately. backend='mlx' (Metal) or 'mock' "
        "(GPU-free, end-to-end). Poll the returned status.json (queued→running→completed|failed).",
        "returns": "JobHandle(run_id, job_dir, status_path, bundle_hdf5, results_path, pid). Results land "
        "at job_dir/outputs/result.hdf5 when status='completed'.",
    },
    "sim_postproc": {
        "name": "sim_postproc",
        "domain": "fdtd",
        "solver": "fdtdmex",
        "import": ("fdtdmex.io", "sim_postproc"),
        "summary": "Reduce a results HDF5 to the small JSON-serializable quantities the agent reads "
        "(per-detector shape/dtype/max_abs/mean_abs/sum + tiny previews). Never returns large field arrays.",
        "returns": "dict: {num_steps, backend, detectors: {name: {key: {shape, dtype, max_abs, mean_abs, ...}}}}.",
    },
    "compute_mode": {
        "name": "compute_mode",
        "domain": "mode",
        "solver": "fdtdmex",
        "import": ("fdtdx.core.physics.modes", "compute_mode"),
        "summary": "Native full-vectorial waveguide mode solver (Tidy3D-free; mode_backend='fdtdmex' is "
        "the default). Computes optical modes of a cross-section from its inverse-permittivity distribution.",
        "returns": "tuple (E, H, complex effective index) jax arrays for the selected mode_index / filter_pol.",
    },
}


# --------------------------------------------------------------------------- #
# Live introspection for get_api_schema (drift-proof against the real functions)
# --------------------------------------------------------------------------- #
def _live_signature(api: dict) -> str | None:
    """``<name><inspect.signature>`` of the real function, or None if it can't be imported."""
    mod_name, attr = api["import"]
    try:
        fn = getattr(importlib.import_module(mod_name), attr)
        return f"{api['name']}{inspect.signature(fn)}"
    except Exception:
        return None


def _scene_model_fields() -> list[str]:
    """The fdtdmex.io.SceneModel field names (pack payload), live. [] on failure."""
    try:
        from fdtdmex.io import SceneModel  # type: ignore

        return list(SceneModel.model_fields.keys())
    except Exception:
        return []


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
    sig = _live_signature(api)
    if sig:
        lines += ["signature (live):", f"  {sig}", ""]
    if name == "pack":
        fields = _scene_model_fields()
        if fields:
            lines += [f"pack payload (fdtdmex.io.SceneModel fields): {', '.join(fields)}", ""]
    lines += [
        f"returns: {api['returns']}",
        "",
        "Next: write native fdtdmex Python — assemble a Scene → bundle = pack(scene, location) → "
        "job = run_simulation_from_hdf5(bundle, parent_folder, backend=...) (non-blocking; poll "
        "job.status_path) → sim_postproc(job.results_path). search_docs(<query>) for a verified example.",
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
        runs_in_proc = "run_fdtd" in doc["body"] and "run_simulation_from_hdf5" not in doc["body"]
        steer = (
            "Next: REUSE this only for the SCENE-BUILDING (geometry/materials/sources/detectors/"
            "compute_mode). This script runs fdtdx IN-PROCESS via run_fdtd/apply_params — in ag-fdtd "
            "that is FORBIDDEN; do NOT copy the run section. To execute, REWRITE it as: "
            "pack(scene, '.') → run_simulation_from_hdf5(bundle, 'jobs', simulation_name=...) (detached, "
            "non-blocking) → read outputs/result.hdf5 (sim_postproc for scalars, h5py for full fields)."
            if runs_in_proc else
            "Next: adapt this setup — assemble the fdtdx.Scene, then pack(scene, '.') → "
            "run_simulation_from_hdf5(bundle, 'jobs', ...) → read outputs/result.hdf5 "
            "(sim_postproc for scalars, h5py for full fields). NEVER run fdtdx in-process (run_fdtd)."
        )
        lines += ["", steer]
    else:
        lines += ["", "Next: search_docs(<query>) for more, or list_solver_apis for the run API."]
    return "\n".join(lines)
