"""Regenerate ``mpb_tilted_uniaxial_slab.npz``: MPB's answer for a tilted-uniaxial slab.

MPB is not a dependency of this repository. Run this once, in an environment that has it, and
commit the resulting ``.npz``::

    micromamba create -y -p <prefix> -c conda-forge python=3.12 pymeep
    <prefix>/bin/python generate_mpb_tilted_uniaxial_slab.py mpb_tilted_uniaxial_slab.npz

The structure is a 400 nm slab of a uniaxial crystal (n_o = 2.0, n_e = 2.2) whose optic axis is
tilted out of the cross-section plane, in an isotropic cladding of index 1.44, inside a 6 um
supercell, at a vacuum wavelength of 1.55 um. MPB solves ``omega(k)``, so the guided index comes
from ``find_k``, which inverts that at the target frequency.

Recorded settings (they belong with the numbers): pymeep 1.34.0 / mpb 1.12.0 / libctl 4.7.1 from
conda-forge, osx-arm64, no MPI; ``mpb.ModeSolver`` with ``tolerance=1e-9``, the default
``mesh_size=3`` dielectric averaging, ``num_bands=2``, ``mp.NO_PARITY``, ``find_k`` tolerance 1e-8
and a k bracket of ``[1.44, 2.31] * omega``; resolutions 200 and 400 pixels per um, i.e. 5 nm and
2.5 nm cells.

A regeneration does not reproduce the committed file bit for bit: MPB's block-iterative eigensolver
starts from a random vector, and re-running this script moved the indices by up to 2.3e-10
(measured). That is four orders below the tolerance the test compares at, but it is the reason the
file is committed rather than regenerated in CI.
"""

import sys

import meep as mp
import numpy as np
from meep import mpb

LAM_UM = 1.55
FREQ = 1.0 / LAM_UM
D_CORE = 0.4
N_CLAD = 1.44
N_O, N_E = 2.0, 2.2
SUPERCELL = 6.0
CASES = ((45.0, 0.0), (55.0, 35.0), (0.0, 0.0))
RESOLUTIONS = (200, 400)
TOLERANCE = 1e-9
NUM_BANDS = 2


def uniaxial(n_o, n_e, axis):
    s = np.asarray(axis, dtype=float)
    s = s / np.linalg.norm(s)
    return n_o**2 * np.eye(3) + (n_e**2 - n_o**2) * np.outer(s, s)


def run(theta_deg, phi_deg, resolution):
    """Effective indices of the guided bands at 1.55 um.

    MPB's z is the propagation direction, x the slab normal, y the invariant direction.
    """
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    eps = uniaxial(N_O, N_E, (np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)))
    medium = mp.Medium(
        epsilon_diag=mp.Vector3(eps[0, 0], eps[1, 1], eps[2, 2]),
        epsilon_offdiag=mp.Vector3(eps[0, 1], eps[0, 2], eps[1, 2]),
    )
    solver = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(SUPERCELL, 0, 0)),
        geometry=[mp.Block(size=mp.Vector3(D_CORE, mp.inf, mp.inf), center=mp.Vector3(), material=medium)],
        resolution=resolution,
        num_bands=NUM_BANDS,
        default_material=mp.Medium(index=N_CLAD),
        tolerance=TOLERANCE,
    )
    ks = solver.find_k(
        mp.NO_PARITY,
        FREQ,
        1,
        NUM_BANDS,
        mp.Vector3(0, 0, 1),
        1e-8,
        N_O * FREQ,
        FREQ * N_CLAD,
        FREQ * N_E * 1.05,
    )
    return [float(k) / FREQ for k in ks]


def main(path):
    out = {}
    for theta, phi in CASES:
        for resolution in RESOLUTIONS:
            key = f"theta{theta:g}_phi{phi:g}_res{resolution}"
            out[key] = np.asarray(run(theta, phi, resolution), dtype=np.float64)
            print(key, out[key], flush=True)
    np.savez(
        path,
        cases=np.asarray(CASES, dtype=np.float64),
        resolutions=np.asarray(RESOLUTIONS, dtype=np.int64),
        wavelength_um=LAM_UM,
        core_thickness_um=D_CORE,
        supercell_um=SUPERCELL,
        n_clad=N_CLAD,
        n_o=N_O,
        n_e=N_E,
        tolerance=TOLERANCE,
        meep_version=mp.__version__,
        **out,
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mpb_tilted_uniaxial_slab.npz")
