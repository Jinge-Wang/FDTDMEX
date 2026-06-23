"""Validation-suite conftest: pin phasor DFT to every-step sampling.

The element-wise oracle tests compare the MLX phasor running-DFT against the JAX run that samples
*every* step. The MLX backend auto-subsamples the phasor DFT by default (a numerical change validated
by physics, not element-wise), so force the exact every-step mode here via ``FDTDMEX_DFT_STRIDE=1``.
``setdefault`` lets an explicit outer override (or a per-test monkeypatch) still win.
"""

import os

os.environ.setdefault("FDTDMEX_DFT_STRIDE", "1")
