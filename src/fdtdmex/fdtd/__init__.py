"""WS-A — the forward FDTD engine.

Curl (spacing-weighted), E/H update (isotropic/diagonal fast path + full-anisotropic 3x3 path),
CPML auxiliary updates, and the Python + ``mx.compile`` time loop. Updates are functional /
out-of-place (race-free). Ports from ../fdtdx (see docs/porting-notes.md, docs/physics.md).

Status: stub.
"""
