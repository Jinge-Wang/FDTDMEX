"""FDTDMEX — forward-first, Metal-native FDTD electromagnetics on Apple Silicon (MLX).

This package is currently a **pre-implementation scaffold**: subpackages define the intended
structure with docstrings and ``NotImplementedError`` stubs. See ``docs/`` for the architecture,
physics conventions, and roadmap, and ``CLAUDE.md`` for agent/contributor guidance.

Subpackages
-----------
backend     MLX helpers (dtype/device, ``mx.compile`` wrappers, complex, np<->mx bridge).
core        config schema, constants, grid (uniform + non-uniform cell-size arrays), typing.
fdtd        WS-A forward engine: curl, E/H update, CPML, time loop.
materials   material model + dispersion (ADE); ``materials.smoothing`` = WS-C subpixel smoothing.
geometry    shapes, GDS import, voxelization, continuous-epsilon evaluator (feeds smoothing).
sources     plane / dipole / TFSF + mode injection.
detectors   field / energy / Poynting / phasor + mode overlap.
modes       WS-B finite-difference mode solver + overlap.
io          (de)serialization: pydantic config <-> JSON, HDF5 results, FDTDX array-bridge.
viz         plotting / export (matplotlib, plotly; pyvista/trame for 3D/web).
"""

from fdtdmex._version import __version__

__all__ = ["__version__"]
