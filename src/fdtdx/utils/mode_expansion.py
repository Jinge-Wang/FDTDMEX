"""Mode-expansion monitor — project a recorded field onto waveguide modes (the PIC transmission view).

Given the complex field at a monitor plane (a :class:`~fdtdx.ModeOverlapDetector`), this projects it
onto a user-specified set of waveguide modes (TE₀, TE₁, TM₀, …) and reports the **power transmitted
into each mode**. It is the FDTDMEX equivalent of a Lumerical mode-expansion monitor / MEEP eigenmode
decomposition.

Each coefficient ``alpha_m`` is the cross-section overlap integral of mode ``m``'s (E, H) with the
recorded field (see :meth:`fdtdx.ModeOverlapDetector.compute_overlap_to_mode`). Because the solver
returns Poynting-normalized modes, the power transmitted into mode ``m`` is
``T_m = |alpha_m / alpha_in|**2`` where ``alpha_in`` is the incident-mode overlap at the source. The
complex coefficient ``s_param = alpha_m / alpha_in`` is the scattering parameter into that mode
(``|s|² = T``, ``arg(s)`` = phase). A single-mode waveguide concentrates ``T`` in its fundamental;
higher-order and cross-polar modes appear as their own (small) channels.

**Mode ordering / naming.** The solver sorts modes by **effective index, descending** (the fundamental
has the highest n_eff), and labels each by its **polarization fraction**: a mode is ``TE`` if the
dominant transverse E-component lies along the first transverse axis (``TM`` along the second), with a
50% threshold. ``TE0, TE1, …`` are then the TE-classified modes in n_eff order. This matches
Lumerical/Tidy3D FDE conventions; MEEP instead numbers bands by n_eff and selects polarization via
mirror *parity* (``eig_parity``) rather than a TE/TM label.

The reference modes can be **computed on the fly** (calling the native mode solver) or loaded from a
**cache file** so repeated analyses don't re-solve. The cache stores the cross-section fingerprint
(grid shape, resolution, frequency, direction, and a permittivity hash) and is only reused when that
fingerprint matches the monitor — so a different geometry never silently reuses stale modes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from loguru import logger

from fdtdx.core.physics.modes import compute_mode

# A mode request is a (polarization, mode_index) pair, e.g. ("te", 0).
ModeKey = tuple[str, int]


@dataclass
class ModeChannel:
    """Transmission into a single waveguide mode."""

    pol: str
    index: int
    n_eff: complex
    coeff: complex  # raw overlap coefficient α_m
    transmission: float  # |α_m / α_in|²  (power into this mode, relative to the input)
    power_fraction: float  # |α_m|² / Σ_k |α_k|²  (share among the requested modes)
    s_param: complex = 0j  # the complex scattering coefficient α_m / α_in (|s|² = transmission)

    @property
    def label(self) -> str:
        return f"{self.pol.upper()}{self.index}"

    @property
    def phase(self) -> float:
        """Phase of the complex S-parameter (radians)."""
        import cmath

        return cmath.phase(self.s_param)


@dataclass
class ModeExpansionResult:
    """Result of decomposing a monitored field onto a set of modes."""

    channels: list[ModeChannel]
    frequency: float
    total_transmission: float  # Σ_m T_m over the requested modes
    n_computed: int = 0  # modes solved on the fly this call
    n_cached: int = 0  # modes reused from the cache file

    def table(self) -> str:
        """A small text table (mode, n_eff, transmission, power share)."""
        lines = [f"{'mode':>6} {'n_eff':>8} {'T (|α/α_in|²)':>16} {'power share':>12}"]
        for c in self.channels:
            lines.append(f"{c.label:>6} {c.n_eff.real:>8.3f} {c.transmission:>16.4f} {c.power_fraction:>12.3f}")
        lines.append(f"{'Σ':>6} {'':>8} {self.total_transmission:>16.4f} {1.0:>12.3f}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """JSON-serializable summary (what an agent/front end reads)."""
        return {
            "frequency": self.frequency,
            "total_transmission": self.total_transmission,
            "modes": [
                {
                    "mode": c.label,
                    "n_eff": [c.n_eff.real, c.n_eff.imag],
                    "transmission": c.transmission,
                    "power_fraction": c.power_fraction,
                    "s_param": [c.s_param.real, c.s_param.imag],  # complex scattering coefficient
                }
                for c in self.channels
            ],
        }

    def plot(self, *, value: str = "transmission", filename: str | Path | None = None):
        """Bar chart of per-mode transmission (or power share). Returns a matplotlib ``Figure``."""
        import matplotlib.pyplot as plt

        labels = [c.label for c in self.channels]
        if value == "power_fraction":
            heights, ylabel, title = [c.power_fraction for c in self.channels], "power share", "Modal power share"
        else:
            heights, ylabel, title = [c.transmission for c in self.channels], "T = |α / α_in|²", "Modal transmission"
        colors = ["tab:blue" if c.pol == "te" else "tab:red" for c in self.channels]
        fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(labels), 3.2))
        ax.bar(labels, heights, color=colors)
        for x, h in enumerate(heights):
            ax.text(x, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} @ {self.frequency / 1e12:.1f} THz")
        ax.set_ylim(0, max(heights) * 1.25 + 1e-9)
        ax.margins(x=0.1)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=200)
            plt.close(fig)
        return fig


# --------------------------------------------------------------------------------------------------
# Mode cache (compute once, reuse across analyses)
# --------------------------------------------------------------------------------------------------


def _cross_section_fingerprint(inv_eps_slice, resolution: float, frequency: float, direction: str) -> dict:
    """A fingerprint identifying the monitor cross-section the cached modes belong to."""
    arr = np.ascontiguousarray(np.asarray(inv_eps_slice, dtype=np.float64).round(10))
    return {
        "shape": list(arr.shape),
        "resolution": round(float(resolution), 12),
        "frequency": round(float(frequency), 3),
        "direction": str(direction),
        "eps_sha1": hashlib.sha1(arr.tobytes()).hexdigest(),
    }


def _load_mode_cache(path: Path, fingerprint: dict) -> dict[ModeKey, tuple] | None:
    """Load cached modes iff the file's fingerprint matches ``fingerprint`` (else ``None``)."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["__meta__"].item() if data["__meta__"].ndim == 0 else data["__meta__"]))
    # "Same domain at least": shape + resolution must match; frequency/direction/eps guard correctness.
    for field in ("shape", "resolution", "frequency", "direction", "eps_sha1"):
        if meta.get(field) != fingerprint.get(field):
            logger.warning(
                f"mode cache {path.name}: {field} mismatch "
                f"(cached {meta.get(field)!r} vs monitor {fingerprint.get(field)!r}) — recomputing."
            )
            return None
    modes: dict[ModeKey, tuple] = {}
    for key in json.loads(str(data["__keys__"].item())):
        pol, idx = key[0], int(key[1])
        p = f"{pol}_{idx}_"
        modes[(pol, idx)] = (jnp.asarray(data[p + "E"]), jnp.asarray(data[p + "H"]), complex(data[p + "neff"].item()))
    return modes


def _save_mode_cache(path: Path, fingerprint: dict, modes: dict[ModeKey, tuple]) -> None:
    arrays = {"__meta__": np.asarray(json.dumps(fingerprint)), "__keys__": np.asarray(json.dumps(list(modes.keys())))}
    for (pol, idx), (E, H, neff) in modes.items():
        p = f"{pol}_{idx}_"
        arrays[p + "E"] = np.asarray(E)
        arrays[p + "H"] = np.asarray(H)
        arrays[p + "neff"] = np.asarray(complex(neff))
    np.savez(path, **arrays)


# --------------------------------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------------------------------


def compute_mode_expansion(
    detector,
    state,
    arrays,
    config,
    modes: Sequence[ModeKey],
    *,
    input_overlap: complex,
    freq_index: int = 0,
    direction: str | None = None,
    cache_path: str | Path | None = None,
) -> ModeExpansionResult:
    """Decompose the field recorded at ``detector`` onto ``modes`` and return per-mode transmission.

    Args:
        detector: A placed :class:`~fdtdx.ModeOverlapDetector` at the monitor plane.
        state: That detector's :class:`DetectorState` from the run (holds the phasor field).
        arrays: The run's :class:`ArrayContainer` (for the cross-section permittivity).
        config: The :class:`SimulationConfig`.
        modes: Requested modes as ``(pol, index)`` pairs, e.g. ``[("te", 0), ("te", 1), ("tm", 0)]``.
        input_overlap: The incident-mode overlap ``α_in`` at the source (normalizes transmission).
        freq_index: Which frequency in the detector's ``wave_characters`` to analyze.
        direction: Mode propagation direction (defaults to the detector's ``direction``).
        cache_path: Optional ``.npz`` path; modes are loaded from it when the cross-section matches,
            and (re)written otherwise — so repeated analyses skip the mode solve.

    Returns:
        A :class:`ModeExpansionResult` with per-mode ``transmission`` and ``power_fraction``.
    """
    direction = direction or detector.direction
    freq = detector.wave_characters[freq_index].get_frequency()
    resolution = detector._mode_solver_resolution()

    inv_eps_slice = arrays.inv_permittivities[:, *detector.grid_slice]
    inv_mu = arrays.inv_permeabilities
    if isinstance(inv_mu, jnp.ndarray) and inv_mu.ndim > 0:
        inv_mu_slice = inv_mu[:, *detector.grid_slice]
    else:
        inv_mu_slice = inv_mu

    fingerprint = _cross_section_fingerprint(inv_eps_slice, resolution, freq, direction)
    path = Path(cache_path) if cache_path is not None else None
    cached = _load_mode_cache(path, fingerprint) if path is not None else None
    if cached is not None:
        logger.info(f"mode expansion: reusing {len(cached)} cached mode(s) from {path.name}")

    transverse_coords = detector._transverse_edge_coordinates()
    solved: dict[ModeKey, tuple] = dict(cached) if cached else {}
    channels: list[ModeChannel] = []
    n_computed = 0
    for pol, idx in modes:
        if (pol, idx) in solved:
            E_m, H_m, neff = solved[(pol, idx)]
        else:
            E_m, H_m, neff = compute_mode(
                frequency=freq,
                inv_permittivities=inv_eps_slice,
                inv_permeabilities=inv_mu_slice,
                resolution=resolution,
                direction=direction,
                mode_index=idx,
                filter_pol=pol,
                dtype=config.dtype,
                transverse_coords=transverse_coords,
            )
            solved[(pol, idx)] = (E_m, H_m, complex(neff))
            n_computed += 1
        alpha = complex(detector.compute_overlap_to_mode(state, E_m, H_m, freq_index))
        channels.append(
            ModeChannel(
                pol=pol,
                index=idx,
                n_eff=complex(neff),
                coeff=alpha,
                transmission=abs(alpha / input_overlap) ** 2,
                power_fraction=0.0,  # filled below once we know the total
                s_param=alpha / input_overlap,  # complex scattering coefficient (magnitude + phase)
            )
        )

    total_power = sum(abs(c.coeff) ** 2 for c in channels) or 1.0
    for c in channels:
        c.power_fraction = abs(c.coeff) ** 2 / total_power

    if path is not None and n_computed > 0:
        _save_mode_cache(path, fingerprint, solved)
        logger.info(f"mode expansion: cached {len(solved)} mode(s) → {path.name}")

    return ModeExpansionResult(
        channels=channels,
        frequency=freq,
        total_transmission=sum(c.transmission for c in channels),
        n_computed=n_computed,
        n_cached=len(modes) - n_computed,
    )
