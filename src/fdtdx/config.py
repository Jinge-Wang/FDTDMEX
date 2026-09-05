import math
import os
from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx import constants
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid, UniformGrid
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.interfaces.recorder import Recorder
from fdtdx.typing import BackendOption


@autoinit
class GradientConfig(TreeClass):
    """Configuration for gradient computation in simulations.

    This class handles settings for automatic differentiation, supporting either
    invertible differentiation with a recorder or checkpointing-based differentiation.

    """

    #: Method for gradient computation.
    #: Can be either "reversible" when using the time reversible autodiff, or "checkpointed" for the exact checkpointing algorithm.
    method: Literal["reversible", "checkpointed"] = frozen_field(default="reversible")

    #: Optional recorder for invertible differentiation. Needs to be provided for reversible autodiff. Defaults to None
    recorder: Recorder | None = field(default=None)

    #: Optional number of checkpoints for checkpointing-based differentiation.
    #: Needs to be provided for checkpointing gradient computation. Defaults to None.
    num_checkpoints: int | None = frozen_field(default=None)

    #: Number of interior full-field checkpoints for the ``"reversible"`` method.
    #: The reversible backward pass reconstructs the field state by running the simulation in
    #: reverse; for lossy materials this reverse reconstruction can accumulate numerical
    #: error over the full trajectory. Setting this to ``k - 1`` partitions the run into ``k`` slices
    #: and stores a full-field checkpoint at each interior slice boundary during the forward pass. The
    #: backward pass then resets the reverse reconstruction to the exact checkpoint at every boundary,
    #: bounding the reconstruction drift to a single slice (``~time_steps_total / k`` steps) at the
    #: cost of O(k) field memory. The default ``0`` reproduces the classic single full reverse pass
    #: (no interior checkpoints; only the final field, which is available for free, is used). Ignored
    #: by the ``"checkpointed"`` method. Must not exceed ``time_steps_total - 1``.
    num_checkpoints_reversible: int = frozen_field(default=0)

    def __post_init__(self):
        if self.method == "reversible" and self.recorder is None:
            raise Exception("Need Recorder in gradient config to compute reversible gradients")
        if self.method == "checkpointed" and self.num_checkpoints is None:
            raise Exception("Need Checkpoint Number in gradient config to compute checkpointed gradients")
        if self.num_checkpoints_reversible < 0:
            raise Exception("num_checkpoints_reversible must be >= 0")


#: The accepted values of :attr:`SimulationConfig.material_sampling`, in widening order.
MATERIAL_SAMPLING_MODES: tuple[str, ...] = ("box", "yee", "yee_smooth")

#: The subset of :data:`MATERIAL_SAMPLING_MODES` that samples per Yee component position rather
#: than once per cell centre. Test membership through
#: :attr:`SimulationConfig.uses_yee_material_sampling`, never by comparing the string literal:
#: a literal ``== "yee"`` silently excludes ``"yee_smooth"``, which is how the source gates in
#: ``linear_polarization.py`` and ``tfsf_region.py`` first went wrong.
YEE_MATERIAL_SAMPLING_MODES: tuple[str, ...] = ("yee", "yee_smooth")

#: Environment variable that turns the box-vs-Yee sampling diagnostic on without touching the config.
YEE_DIAGNOSTICS_ENV_VAR = "FDTDX_YEE_SAMPLING_DIAGNOSTICS"

_TRUTHY = ("1", "true", "yes", "on")


@autoinit
class SimulationConfig(TreeClass):
    """Configuration settings for FDTD simulations.

    This class contains all the parameters needed to configure and run an FDTD
    simulation, including spatial and temporal discretization, hardware backend,
    and gradient computation settings.

    """

    #: Total simulation time in seconds.
    time: float = frozen_field()

    #: Spatial grid configuration.
    #:
    #: ``UniformGrid`` is an unresolved policy used while the final volume shape
    #: is still being inferred.  ``RectilinearGrid`` is the realized solver grid
    #: with explicit physical edge coordinates.  Placement resolves policies to
    #: ``RectilinearGrid`` so compiled FDTD code has exactly one metric source.
    grid: UniformGrid | QuasiUniformGrid | RectilinearGrid = field()

    #: Computation backend ('gpu', 'tpu', 'cpu' or 'METAL'). Defaults to "gpu".
    backend: BackendOption = frozen_field(default="gpu")

    #:  Data type for numerical computations. Defaults to jnp.float32.
    dtype: jnp.dtype = frozen_field(default=jnp.float32)

    #: Whether to use complex-valued field arrays.
    #: None (default): auto-detect based on boundary conditions (e.g. Bloch).
    #: True: force complex fields (complex64 if dtype=float32, complex128 if dtype=float64).
    #: False: force real fields (raises error if Bloch boundaries are present).
    use_complex_fields: bool | None = frozen_field(default=None)

    #: Safety factor for the Courant condition (default: 0.99).
    courant_factor: float = frozen_field(default=0.99)

    #: Per-axis mirror symmetry of the simulation, in the order (x, y, z).
    #: Each entry is one of ``{-1, 0, +1}``:
    #: ``0`` = no symmetry on this axis (default),
    #: ``-1`` = PEC (electric-wall) mirror on the axis center plane,
    #: ``+1`` = PMC (magnetic-wall) mirror on the axis center plane.
    #: When any entry is nonzero, :func:`fdtdx.place_objects` automatically reduces the
    #: domain to the symmetric half/quarter/octant (keeping the upper half along each
    #: symmetric axis) and clips every object onto that reduced grid. An electric plane
    #: lands on the reduced domain's min edge and gets a PEC wall there; a magnetic plane
    #: sits half a cell below it (sources and materials are rasterized per cell), where the
    #: zero field halo already is the exact mirror, so it gets no wall object. Mode sources
    #: and mode-overlap detectors solve on the mirrored full cross-section and restrict,
    #: rather than using the mode solver's own symmetric solve. The FDTD then runs on the
    #: reduced domain; call
    #: :func:`fdtdx.unfold_fields` / :func:`fdtdx.unfold_detector_states` afterwards to
    #: reconstruct the full-domain arrays. This is additive and independent of manually
    #: specifying PEC/PMC as ordinary boundaries via :class:`fdtdx.BoundaryConfig`.
    #: Each symmetric axis must resolve to an **even** number of grid cells (so the domain
    #: splits exactly down the middle and the unfolded result matches the full domain
    #: cell-for-cell); otherwise :func:`fdtdx.place_objects` raises a ``ValueError``.
    symmetry: tuple[int, int, int] = frozen_field(default=(0, 0, 0))

    #: Where the material of a static object is sampled when the arrays are assembled.
    #: ``"box"`` (default, legacy) samples every object once per cell centre and broadcasts the
    #: single mask to all field components, after the object's extent has been rounded to a whole
    #: number of cells. ``"yee"`` keeps every static object's requested metric extent continuous and
    #: samples the material once per Yee component position (E_x, E_y, E_z, and the H positions when
    #: a permeability or magnetic-conductivity array exists), taking the material of the
    #: highest-priority object that contains that point. Priority is the order in which objects are
    #: written today (``placement_order`` ascending, later wins), with the simulation volume as the
    #: background. Devices, sources, detectors and PML keep their integer boxes in both modes.
    #: ``"yee_smooth"`` is ``"yee"`` plus a Kottke/Farjadpour sub-pixel post-pass: at every pixel
    #: where exactly two materials meet, the point sample is replaced by the effective inverse
    #: permittivity of that pixel (harmonic mean along the interface normal, arithmetic mean in the
    #: interface plane), which removes the first-order staircasing error. Uniform pixels and pixels
    #: holding three or more materials keep their point sample; conductivity and dispersion are never
    #: averaged. Note that ``"yee_smooth"`` is not bit-identical to ``"box"`` with object-level
    #: ``subpixel_smoothing`` for tilted interfaces: it takes the diagonal of the *inverse* effective
    #: tensor, which is the entry the elementwise update applies to ``E_c``.
    material_sampling: Literal["box", "yee", "yee_smooth"] = frozen_field(default="box")

    #: Samples per axis used by ``material_sampling="yee_smooth"`` for a pixel fill fraction or an
    #: interface normal that the object's shape cannot answer analytically (a sphere, a tapered
    #: sidewall). Boxes, cylinders and polygon extrusions are exact and never use it.
    yee_smooth_supersample: int = frozen_field(default=8)

    #: Keep the off-diagonal Kottke terms under ``material_sampling="yee_smooth"``, allocating the
    #: full 9-component inverse permittivity tensor. More accurate for tilted interfaces and
    #: identical to the default diagonal tier for axis-aligned ones, but it costs the Metal
    #: block-hybrid kernel wherever the tilted pixels are scattered.
    yee_smooth_full_tensor: bool = frozen_field(default=False)

    #: Report how many Yee sample points disagree with what the legacy ``"box"`` path would have
    #: written, in ``info["yee_sampling_difference"]``. Off by default: answering it rasterises the
    #: whole scene a second time on the cell-centre lattice and holds another ``int32`` copy of the
    #: domain, and nothing in the simulation reads the answer. The measured cost is scene-dependent
    #: and smaller than a doubling — about 8% on a 120 x 120 x 30 grid with three nested cylinders,
    #: because the comparison pass only touches each object's rounded box while the Yee pass
    #: evaluates three full-domain lattices — but it is pure diagnostic work either way. The
    #: environment variable ``FDTDX_YEE_SAMPLING_DIAGNOSTICS=1`` turns it on without editing the
    #: config. The smoothing counters under ``info["yee_sampling_difference"]["smoothing"]`` come
    #: out of the pass that has to run anyway and are always reported.
    yee_sampling_diagnostics: bool = frozen_field(default=False)

    #: Optional configuration for gradient computation.
    gradient_config: GradientConfig | None = field(default=None)

    def __post_init__(self):
        from jax import extend

        if self.material_sampling not in MATERIAL_SAMPLING_MODES:
            raise ValueError(
                f"config.material_sampling must be 'box', 'yee' or 'yee_smooth', got {self.material_sampling!r}"
            )

        if self.yee_smooth_supersample < 1:
            raise ValueError(f"config.yee_smooth_supersample must be >= 1, got {self.yee_smooth_supersample}")

        if len(self.symmetry) != 3 or any(s not in (-1, 0, 1) for s in self.symmetry):
            raise ValueError(
                f"config.symmetry must be a length-3 tuple with each entry in {{-1, 0, +1}} "
                f"(0=none, -1=PEC, +1=PMC), got {self.symmetry!r}"
            )

        current_platform = extend.backend.get_backend().platform

        if current_platform == "METAL" and self.backend == "gpu":
            self.backend = "METAL"

        if self.backend == "METAL":
            try:
                jax.devices()
                if __name__ == "__main__":
                    logger.info("METAL device found and will be used for computations")
                jax.config.update("jax_platform_name", "metal")
            except RuntimeError:
                if __name__ == "__main__":
                    logger.warning("METAL initialization failed, falling back to CPU!")
                self.backend = "cpu"
        elif self.backend in ["gpu", "tpu"]:
            try:
                jax.devices(self.backend)
                if __name__ == "__main__":
                    logger.info(f"{str.upper(self.backend)} found and will be used for computations")
                jax.config.update("jax_platform_name", self.backend)
            except RuntimeError:
                if __name__ == "__main__":
                    logger.warning(f"{str.upper(self.backend)} not found, falling back to CPU!")
                self.backend = "cpu"

        if self.backend == "cpu":
            jax.config.update("jax_platform_name", "cpu")

    @property
    def has_symmetry(self) -> bool:
        """Whether any axis requests mirror symmetry.

        Returns:
            bool: True if at least one entry of :attr:`symmetry` is nonzero, meaning the
                domain will be reduced and a PEC/PMC wall placed on the symmetry plane(s).
        """
        return any(s != 0 for s in self.symmetry)

    @property
    def uses_yee_material_sampling(self) -> bool:
        """Whether static materials are sampled at the Yee component positions.

        True for both ``material_sampling="yee"`` and ``material_sampling="yee_smooth"``. This is
        the predicate every caller should use: the two modes share one lattice, one priority rule
        and one set of array-tier consequences, and they differ only in what happens afterwards at
        the two-material pixels.

        Returns:
            bool: True when the per-Yee-point loader assembles the static arrays.
        """
        return self.material_sampling in YEE_MATERIAL_SAMPLING_MODES

    @property
    def uses_yee_smoothing(self) -> bool:
        """Whether the Kottke sub-pixel post-pass runs on top of the Yee sampling.

        Returns:
            bool: True only for ``material_sampling="yee_smooth"``.
        """
        return self.material_sampling == "yee_smooth"

    @property
    def yee_sampling_diagnostics_enabled(self) -> bool:
        """Whether to spend a second rasterisation on the box-vs-Yee difference diagnostic.

        Returns:
            bool: True when :attr:`yee_sampling_diagnostics` is set or the environment variable
                ``FDTDX_YEE_SAMPLING_DIAGNOSTICS`` is one of ``1``/``true``/``yes``/``on``.
        """
        if self.yee_sampling_diagnostics:
            return True
        return os.environ.get(YEE_DIAGNOSTICS_ENV_VAR, "").strip().lower() in _TRUTHY

    @property
    def courant_number(self) -> float:
        """Calculate the Courant number for the simulation.

        The Courant number is a dimensionless quantity that determines stability
        of the FDTD simulation. It represents the ratio of the physical propagation
        speed to the numerical propagation speed.

        Returns:
            float: The Courant number, scaled by the courant_factor and normalized
                for 3D simulations.
        """
        return self.courant_factor / math.sqrt(3)

    def resolve_grid(self, shape: tuple[int, int, int] | None = None) -> RectilinearGrid:
        """Return a concrete solver grid.

        Args:
            shape: Required when ``grid`` is an unresolved ``UniformGrid``.

        Returns:
            A concrete ``RectilinearGrid``.
        """
        if isinstance(self.grid, RectilinearGrid):
            return self.grid
        if shape is None:
            raise ValueError("A grid shape is required to resolve UniformGrid.")
        return self.grid.resolve(shape)

    @property
    def resolved_grid(self) -> RectilinearGrid | None:
        """Return the concrete solver grid, or ``None`` if not yet resolved.

        ``UniformGrid`` has no edge arrays until the simulation shape is known.
        Callers that need coordinates, areas, or volumes should use this
        property and fall back to ``uniform_spacing`` when it returns ``None``.
        """
        if isinstance(self.grid, RectilinearGrid):
            return self.grid
        return None

    @property
    def has_nonuniform_grid(self) -> bool:
        """Whether the realized solver grid is non-uniform."""
        grid = self.resolved_grid
        return grid is not None and not grid.is_uniform

    def uniform_spacing(self) -> float:
        """Return the uniform grid spacing.

        ``UniformGrid`` can answer this before placement.  ``RectilinearGrid``
        answers only when all spacings are equal and raises for non-uniform
        meshes, making unsupported scalar assumptions explicit.
        """
        if isinstance(self.grid, UniformGrid):
            return self.grid.spacing
        if isinstance(self.grid, QuasiUniformGrid):
            if self.grid.is_uniform:
                return self.grid.dx
            else:
                raise ValueError(
                    "QuasiUniformGrid has no single uniform spacing:"
                    f" ({self.grid.dx}, {self.grid.dy}, {self.grid.dz} differ). "
                )
        return self.grid.uniform_spacing  # RectilinearGrid — raises internally if non-uniform

    @property
    def time_step_duration(self) -> float:
        """Calculate the duration of a single time step.

        The time step duration is determined by the Courant condition to ensure
        numerical stability. Realized rectilinear grids use their smallest
        per-axis spacings. Unresolved uniform grids use their configured scalar
        spacing; unresolved quasi-uniform grids use their smallest per-axis
        spacing as a conservative CFL bound.

        Returns:
            float: Time step duration in seconds, calculated using the Courant
                condition and spatial resolution.
        """
        if isinstance(self.grid, RectilinearGrid):
            return self.grid.cfl_time_step(self.courant_factor)
        if isinstance(self.grid, UniformGrid):
            return self.courant_number * self.grid.spacing / constants.c
        if isinstance(self.grid, QuasiUniformGrid):
            return self.courant_number * self.grid.min_spacing / constants.c
        raise NotImplementedError(f"time_step_duration is not implemented for grid type {type(self.grid).__name__}.")

    @property
    def time_steps_total(self) -> int:
        """Calculate the total number of time steps for the simulation.

        Determines how many discrete time steps are needed to simulate the
        specified total simulation time, based on the time step duration.

        Returns:
            int: Total number of time steps needed to reach the specified
                simulation time.
        """
        return round(self.time / self.time_step_duration)

    @property
    def max_travel_distance(self) -> float:
        """Calculate the maximum distance light can travel during the simulation.

        This represents the theoretical maximum distance that light could travel
        through the simulation volume, useful for determining if the simulation
        time is sufficient for light to traverse the entire domain.

        Returns:
            float: Maximum travel distance in meters, based on the speed of light
                and total simulation time.
        """
        return constants.c * self.time

    @property
    def only_forward(self) -> bool:
        """Check if the simulation is forward-only (no gradient computation).

        Forward-only simulations don't compute gradients and are used when only
        the forward propagation of electromagnetic fields is needed, without
        optimization.

        Returns:
            bool: True if no gradient configuration is specified, False otherwise.
        """
        return self.gradient_config is None

    @property
    def invertible_optimization(self) -> bool:
        """Check if invertible optimization is enabled.

        Invertible optimization uses time-reversibility of Maxwell's equations
        to compute gradients with reduced memory requirements compared to
        checkpointing-based methods.

        Returns:
            bool: True if gradient computation uses invertible differentiation
                (recorder is specified), False otherwise.
        """
        if self.gradient_config is None:
            return False
        return self.gradient_config.recorder is not None


DUMMY_SIMULATION_CONFIG = SimulationConfig(
    time=-1,
    grid=UniformGrid(spacing=1),
)
