"""MLX backend helpers.

Centralizes MLX usage so the rest of the package is insulated from API details:
dtype/device selection, ``mx.compile`` wrappers for the per-step kernel, complex-field helpers,
periodic ``mx.eval`` graph-bounding, and the NumPy<->MLX array bridge.

Status: stub. No implementation yet — see docs/getting-started.md and the `porting-from-fdtdx` skill.
"""
