"""Region-restricted detector interpolation is element-wise identical to the full-domain path.

``interpolate_region_mlx`` co-locates E and the time-averaged H onto the E_z point over only a
detector's ``grid_slice`` (+ 1-cell halo). It must reproduce, bit-for-bit, the old hot path:
``interpolate_fields_mlx(pad_fields_mlx(E), pad_fields_mlx((H_prev+H)/2))`` sliced to that region —
across uniform / non-uniform weights and interior / domain-edge / periodic windows.
"""

import numpy as np
import pytest

from fdtdx.backend.platform import is_apple_silicon, mlx_available

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not (is_apple_silicon() and mlx_available()),
        reason="MLX (Metal) backend requires Apple Silicon + mlx",
    ),
]


def _broadcast_axis(arr_1d, axis):
    import mlx.core as mx

    shape = [1, 1, 1]
    shape[axis] = arr_1d.shape[0]
    return mx.array(np.ascontiguousarray(arr_1d)).reshape(shape)


def _nonuniform_widths(shape, rng):
    """Synthetic ``interp_widths`` = per-axis ``(cur_half, prev_half)`` broadcast tables. The
    exactness claim is independent of the actual values, so arbitrary positive widths suffice."""
    widths = []
    for a, n in enumerate(shape):
        cur = (0.5 + rng.random(n)).astype(np.float32)
        prev = (0.5 + rng.random(n)).astype(np.float32)
        widths.append((_broadcast_axis(cur, a), _broadcast_axis(prev, a)))
    return tuple(widths)


def _full_reference(E, H_prev, H_cur, grid_slice, periodic_axes, interp_widths):
    from fdtdx.mlx.curl import pad_fields_mlx
    from fdtdx.mlx.interpolate import interpolate_fields_mlx

    E_pad = pad_fields_mlx(E, periodic_axes)
    H_pad = pad_fields_mlx(0.5 * (H_prev + H_cur), periodic_axes)
    E_i, H_i = interpolate_fields_mlx(E_pad, H_pad, interp_widths)
    sl = (slice(None), *grid_slice)
    return E_i[sl], H_i[sl]


@pytest.mark.parametrize(
    "grid_slice, periodic_axes",
    [
        # interior box
        ((slice(4, 9), slice(5, 7), slice(3, 10)), (False, False, False)),
        # touches the low x edge and the high z edge (zero ghosts)
        ((slice(0, 5), slice(2, 6), slice(8, 12)), (False, False, False)),
        # periodic x with a window on the low edge (wrap ghost), interior elsewhere
        ((slice(0, 4), slice(3, 8), slice(2, 9)), (True, False, False)),
        # full span on every axis == the whole domain
        ((slice(0, 12), slice(0, 10), slice(0, 12)), (False, True, False)),
    ],
)
@pytest.mark.parametrize("uniform", [True, False])
def test_region_interp_matches_full(grid_slice, periodic_axes, uniform):
    import mlx.core as mx

    from fdtdx.mlx.interpolate import interpolate_region_mlx

    shape = (12, 10, 12)
    rng = np.random.default_rng(0)
    E = mx.array(rng.standard_normal((3, *shape)).astype(np.float32))
    H_prev = mx.array(rng.standard_normal((3, *shape)).astype(np.float32))
    H_cur = mx.array(rng.standard_normal((3, *shape)).astype(np.float32))
    interp_widths = None if uniform else _nonuniform_widths(shape, rng)

    E_ref, H_ref = _full_reference(E, H_prev, H_cur, grid_slice, periodic_axes, interp_widths)
    E_reg, H_reg = interpolate_region_mlx(E, H_prev, H_cur, grid_slice, periodic_axes, interp_widths)

    assert E_reg.shape == E_ref.shape
    assert H_reg.shape == H_ref.shape
    # Same arithmetic on the same values -> bit-identical (allow a float32 rounding epsilon).
    assert float(mx.max(mx.abs(E_reg - E_ref))) < 1e-6
    assert float(mx.max(mx.abs(H_reg - H_ref))) < 1e-6
