"""``Scene`` — a thin Tidy3D-like facade over the ``place_objects → apply_params → run_fdtd`` flow.

The low-level API stays intact; this only removes boilerplate so a notebook can::

    sim = Scene(config)
    sim.add(volume, core, source, detector)
    sim.constrain(*constraints)
    sim.plot()              # matplotlib setup view (renders inline)
    data = sim.run()        # MLX auto-routed; same result as the explicit calls
    sim.results             # the returned ArrayContainer

It also bridges to the pydantic config schema (``fdtdmex.io.schema.SceneModel``) via
:meth:`to_model` / :meth:`from_model`, and to the HDF5 hand-off via :meth:`sim_init`.

This facade is deliberately **non-blocking**: it never prompts for confirmation. The agentic
workflow's confirm-before-run gate lives in the orchestration layer (MCP/AG-UI), not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.objects.object import SimulationObject

if TYPE_CHECKING:
    from fdtdx.conversion.json import JsonSetup


class Scene:
    """Bundle ``config + object_list + constraints`` with place / plot / run helpers.

    Args:
        config: The :class:`~fdtdx.config.SimulationConfig` for the run.
        objects: Optional initial objects (also addable later with :meth:`add`).
        constraints: Optional initial placement constraints (also via :meth:`constrain`).
    """

    def __init__(
        self,
        config: SimulationConfig,
        objects: Sequence[SimulationObject] | None = None,
        constraints: Sequence[Any] | None = None,
    ) -> None:
        self.config = config
        self.object_list: list[SimulationObject] = list(objects) if objects else []
        self.constraints: list[Any] = list(constraints) if constraints else []
        # Filled by .place() / .run().
        self.objects: ObjectContainer | None = None
        self.arrays: ArrayContainer | None = None
        self.params: Any | None = None
        self.results: ArrayContainer | None = None

    # ----- assembly -----------------------------------------------------------------------------
    def add(self, *objects: SimulationObject) -> "Scene":
        """Append simulation objects (volume, materials, sources, detectors, boundaries)."""
        for o in objects:
            self.object_list.append(o)
        return self

    def constrain(self, *constraints: Any) -> "Scene":
        """Append placement constraints. Accepts individual constraints or iterables of them."""
        for c in constraints:
            if isinstance(c, (list, tuple)):
                self.constraints.extend(c)
            else:
                self.constraints.append(c)
        return self

    # ----- resolve / run ------------------------------------------------------------------------
    def place(self, key: Any | None = None) -> "Scene":
        """Resolve constraints and initialize arrays (``place_objects`` then ``apply_params``).

        Caches ``objects``, ``arrays``, ``params`` and the resolved ``config``. Idempotent enough
        to call before :meth:`plot` to view the *placed* setup.
        """
        objects, arrays, params, config, _ = place_objects(
            object_list=self.object_list, config=self.config, constraints=self.constraints, key=key
        )
        arrays, objects, _ = apply_params(arrays, objects, params, key)
        self.objects, self.arrays, self.params, self.config = objects, arrays, params, config
        return self

    def run(self, key: Any | None = None, show_progress: bool = False, **run_kwargs: Any) -> ArrayContainer:
        """Place (if needed) and run the forward simulation. Returns the result ``ArrayContainer``.

        MLX auto-routing is handled inside ``run_fdtd``; this matches the explicit
        ``place_objects → apply_params → run_fdtd`` path exactly.
        """
        if self.objects is None or self.arrays is None:
            self.place(key=key)
        assert self.objects is not None and self.arrays is not None
        _, result = run_fdtd(
            arrays=self.arrays,
            objects=self.objects,
            config=self.config,
            key=key,
            show_progress=show_progress,
            **run_kwargs,
        )
        self.results = result
        return result

    @property
    def detector_states(self) -> dict[str, Any] | None:
        """The ``{name: {key: array}}`` detector results from the last :meth:`run`."""
        if self.results is None:
            return None
        return self.results.detector_states

    # ----- visualization ------------------------------------------------------------------------
    def plot(self, **kwargs: Any):
        """Matplotlib setup view (XY/XZ/YZ panels). Wraps :func:`fdtdx.plot_setup`."""
        from fdtdx.utils.plot_setup import plot_setup

        if self.objects is None:
            self.place()
        return plot_setup(config=self.config, objects=self.objects, **kwargs)

    def plot_material(self, **kwargs: Any):
        """Matplotlib material (permittivity) cross-section. Wraps :func:`fdtdx.plot_material`."""
        from fdtdx.utils.plot_material import plot_material

        if self.objects is None or self.arrays is None:
            self.place()
        return plot_material(config=self.config, arrays=self.arrays, **kwargs)

    def plot3d(self, *, confirm: bool = False, **kwargs: Any):
        """Interactive plotly 3D setup view. Wraps :func:`fdtdx.utils.plot_setup_3d.plot_setup_3d`.

        ``confirm=True`` attaches the confirm-required affordance used by the agentic flow; the
        default is a plain, non-blocking figure for manual/notebook use.
        """
        from fdtdx.utils.plot_setup_3d import plot_setup_3d

        if self.objects is None:
            self.place()
        return plot_setup_3d(self.config, self.objects, material_arrays=self.arrays, confirm=confirm, **kwargs)

    # ----- schema / hand-off bridges ------------------------------------------------------------
    def to_json_setup(self) -> "JsonSetup":
        """Return the serializable ``JsonSetup`` (config + object_list + constraints)."""
        from fdtdx.conversion.json import JsonSetup

        return JsonSetup(config=self.config, object_list=self.object_list, constraints=self.constraints)

    @classmethod
    def from_json_setup(cls, setup: "JsonSetup") -> "Scene":
        """Build a :class:`Scene` from a ``JsonSetup``."""
        return cls(config=setup.config, objects=setup.object_list, constraints=setup.constraints)

    def to_model(self):
        """Return the pydantic ``SceneModel`` mirror (requires the ``io`` extra)."""
        from fdtdmex.io.schema import SceneModel

        return SceneModel.from_json_setup(self.to_json_setup())

    @classmethod
    def from_model(cls, model: Any) -> "Scene":
        """Build a :class:`Scene` from a pydantic ``SceneModel`` (requires the ``io`` extra)."""
        return cls.from_json_setup(model.to_json_setup())

    def pack(self, location: Any, **kwargs: Any):
        """Resolve + pack this scene into a project folder as a portable bundle (``io`` extra).

        The agent-facing form: writes a content-addressed config HDF5 + lightweight config JSON into
        ``location`` and returns a ``PackResult``. See :func:`fdtdmex.io.pack`.
        """
        from fdtdmex.io import pack

        return pack(self, location, **kwargs)

    def sim_init(self, path: Any, **kwargs: Any):
        """Resolve + pack this scene into a config HDF5 at an explicit *file* path (``io`` extra).

        Low-level primitive; prefer :meth:`pack` for the folder-owning, content-addressed form.
        """
        from fdtdmex.io import sim_init

        return sim_init(self, path, **kwargs)

    # ----- reprs --------------------------------------------------------------------------------
    def _summary(self) -> dict[str, Any]:
        objs = self.objects.objects if self.objects is not None else self.object_list
        # Counts by base class without importing every concrete type.
        from fdtdx.objects.detectors.detector import Detector
        from fdtdx.objects.sources.source import Source

        sources = [o for o in objs if isinstance(o, Source)]
        detectors = [o for o in objs if isinstance(o, Detector)]
        return {
            "n_objects": len(objs),
            "n_sources": len(sources),
            "n_detectors": len(detectors),
            "n_constraints": len(self.constraints),
            "placed": self.objects is not None,
            "ran": self.results is not None,
            "sources": [o.name for o in sources],
            "detectors": [o.name for o in detectors],
        }

    def __repr__(self) -> str:
        s = self._summary()
        return (
            f"Scene(objects={s['n_objects']}, sources={s['n_sources']}, detectors={s['n_detectors']}, "
            f"constraints={s['n_constraints']}, placed={s['placed']}, ran={s['ran']})"
        )

    def _repr_html_(self) -> str:
        s = self._summary()
        try:
            steps = int(self.config.time_steps_total)
        except Exception:
            steps = None
        rows = [
            ("objects", s["n_objects"]),
            ("sources", f"{s['n_sources']} {s['sources']}"),
            ("detectors", f"{s['n_detectors']} {s['detectors']}"),
            ("constraints", s["n_constraints"]),
            ("time", f"{self.config.time:.3e} s" + (f" ({steps} steps)" if steps else "")),
            ("placed", s["placed"]),
            ("ran", s["ran"]),
        ]
        body = "".join(
            f"<tr><td style='text-align:right;padding-right:8px;color:#888'>{k}</td><td>{v}</td></tr>" for k, v in rows
        )
        return f"<table><caption style='text-align:left'><b>Scene</b></caption>{body}</table>"
