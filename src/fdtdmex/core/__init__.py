"""Core types: configuration schema, physical constants, grid, and typing.

The grid must support **non-uniform** meshes and expose per-axis Yee cell-size arrays
(primal/dual spacings), since every operator is spacing-weighted. See docs/nonuniform-grid.md.

Status: stub.
"""
