"""Serialization and interop.

Declarative pydantic config <-> JSON; HDF5 for large field results; and the FDTDX **array-bridge**
(run FDTDX's CPU front end to obtain material/PML/source arrays, convert NumPy -> MLX). See
docs/mcp-and-ui.md (config schema) and docs/porting-notes.md (array bridge).

Status: stub.
"""
