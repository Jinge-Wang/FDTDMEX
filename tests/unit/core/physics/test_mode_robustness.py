"""The spurious-mode filter: what the discrete operator admits and the structure does not support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.modes import (
    ModeTupleType,
    compute_modes,
    filter_spurious_modes,
    wall_energy_fraction,
)

FREQ = 299792458.0 / 1.55e-6


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _mode(energy_map, neff=1.0):
    field = np.sqrt(np.asarray(energy_map, dtype=float))
    zero = np.zeros_like(field)
    return ModeTupleType(neff=complex(neff), Ex=field, Ey=zero, Ez=zero, Hx=zero, Hy=zero, Hz=zero)


class TestWallEnergyFraction:
    def test_a_centred_mode_leaves_nothing_at_the_walls(self):
        energy = np.zeros((9, 7))
        energy[3:6, 2:5] = 1.0
        assert wall_energy_fraction(_mode(energy)) == pytest.approx(0.0)

    def test_a_mode_living_on_one_wall_column_is_all_ring(self):
        energy = np.zeros((9, 7))
        energy[:, 0] = 1.0
        assert wall_energy_fraction(_mode(energy)) == pytest.approx(1.0)

    def test_a_collapsed_cross_section_has_no_ring(self):
        energy = np.ones((9, 1))
        assert wall_energy_fraction(_mode(energy)) == 0.0


class TestFilterSpuriousModes:
    def test_an_index_above_every_material_is_rejected(self):
        energy = np.zeros((9, 7))
        energy[3:6, 2:5] = 1.0
        modes = [_mode(energy, neff=3.6), _mode(energy, neff=2.0)]
        kept, rejected = filter_spurious_modes(modes, max_material_index=3.48)
        assert [complex(m.neff).real for m in kept] == [2.0]
        assert rejected[0].reason == "index_above_material"
        assert rejected[0].index == 0

    def test_energy_piled_against_the_wall_is_rejected(self):
        centred = np.zeros((9, 7))
        centred[3:6, 2:5] = 1.0
        wall = np.zeros((9, 7))
        wall[0, :] = 1.0
        kept, rejected = filter_spurious_modes([_mode(wall, 1.4), _mode(centred, 1.2)], max_material_index=3.48)
        assert len(kept) == 1
        assert rejected[0].reason == "energy_at_the_walls"
        assert rejected[0].detail == pytest.approx(1.0)

    def test_a_clean_list_is_returned_unchanged(self):
        energy = np.zeros((9, 7))
        energy[3:6, 2:5] = 1.0
        modes = [_mode(energy, neff=n) for n in (3.0, 2.0, 1.6)]
        kept, rejected = filter_spurious_modes(modes, max_material_index=3.48)
        assert kept == modes
        assert rejected == []


class TestFilterOnRealCrossSections:
    """A 40 x 30 strip at 40 nm: the operator's PEC-wall artefacts, and which of them get dropped."""

    @staticmethod
    def _strip():
        eps = np.full((40, 30), 2.25)
        eps[14:26, 10:16] = 12.0
        return jnp.asarray(eps)[None, None, :, :]

    def test_the_default_list_still_carries_wall_modes(self, float64):
        """The filter is opt-in, so the unfiltered list is exactly what it always was."""
        _, _, neff = compute_modes(
            frequency=FREQ,
            inv_permittivities=1.0 / self._strip(),
            inv_permeabilities=1.0,
            num_modes=6,
            resolution=40e-9,
            dtype=jnp.float64,
        )
        values = np.real(np.asarray(neff))
        # Two modes sit at exactly the cladding index: they live on the PEC wall lines, not in the
        # structure, and compute_mode(mode_index=3) hands one of them back today.
        assert np.sum(np.abs(values - 1.5) < 1e-9) == 2

    def test_the_filter_removes_them_and_keeps_the_guided_ones(self, float64):
        _, _, neff = compute_modes(
            frequency=FREQ,
            inv_permittivities=1.0 / self._strip(),
            inv_permeabilities=1.0,
            num_modes=6,
            resolution=40e-9,
            dtype=jnp.float64,
            drop_spurious=True,
        )
        values = np.real(np.asarray(neff))
        assert np.all(np.abs(values - 1.5) > 1e-9)
        assert values[0] == pytest.approx(2.499786, abs=1e-5)
        assert values[1] == pytest.approx(1.967376, abs=1e-5)
        assert values[2] == pytest.approx(1.556305, abs=1e-5)

    def test_a_confined_high_contrast_cross_section_loses_nothing(self, float64):
        """The phase-shifter cross-section has no artefact in its first eight modes; the filter is a
        no-op there, which is the check that it does not eat physical modes."""
        n_si, n_sio2 = 3.48, 1.55
        eps = np.full((120, 48), n_sio2**2)
        eps[40:80, 22:26] = n_si**2
        permittivity = jnp.asarray(eps)[None, :, :, None]
        common = dict(
            frequency=FREQ,
            inv_permittivities=1.0 / permittivity,
            inv_permeabilities=1.0,
            num_modes=6,
            resolution=50e-9,
            dtype=jnp.float64,
        )
        _, _, plain = compute_modes(**common)
        _, _, filtered = compute_modes(**common, drop_spurious=True)
        assert np.allclose(np.asarray(plain), np.asarray(filtered))

    def test_a_metal_cell_loosens_the_index_bound(self, float64):
        """A negative real permittivity means surface modes above every dielectric index, so the
        bound is taken from |eps| and nothing is thrown away for being 'too fast'."""
        n_tin, k_tin = 3.1477, 5.8429
        eps_tin = (n_tin**2 - k_tin**2) + 2j * n_tin * k_tin
        eps = np.full((60, 40), 2.4025, dtype=np.complex128)
        eps[20:40, 16:22] = 12.11
        eps[20:40, 22:25] = eps_tin
        permittivity = jnp.asarray(eps)[None, :, :, None]
        _, _, neff = compute_modes(
            frequency=FREQ,
            inv_permittivities=1.0 / permittivity,
            inv_permeabilities=1.0,
            num_modes=4,
            resolution=50e-9,
            dtype=jnp.float64,
            drop_spurious=True,
        )
        assert np.real(np.asarray(neff))[0] > 3.0
