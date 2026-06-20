"""WS-B — finite-difference waveguide mode solver and overlap.

Assemble a sparse full-vectorial transverse-field operator on a 2D Yee cross-section (consuming the
smoothed material tensors) and solve a few modes with scipy.sparse.linalg.eigs on the host ->
(n_eff, transverse E/H). Mode overlap = spatial integral vs a field monitor. Injection lives in
``fdtdmex.sources``. See docs/mode-solver.md.

Status: stub.
"""
