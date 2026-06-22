"""Serializable S-matrix result + table/heatmap plot (Phase 4 Track A front-end).

Wraps the ``dict[(out_port, in_port) -> complex array]`` returned by
:func:`fdtdx.utils.sparams.calculate_sparams` in a small dataclass that round-trips to JSON (so a
front-end / notebook can read scalar results without touching field arrays) and renders a
magnitude/phase table. The S-matrix *computation* lives in ``sparams.py``; this is presentation only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class SMatrixResult:
    """An S-matrix indexed by ``(output_port, input_port)`` over one or more frequencies.

    Attributes:
        data: maps ``(out_port, in_port)`` to a complex 1-D array of scattering amplitudes, one per
            frequency (the shape produced by :func:`fdtdx.utils.sparams.calculate_sparams`).
        frequencies: optional frequencies (Hz) labelling the array axis.
    """

    data: dict[tuple[str, str], np.ndarray]
    frequencies: list[float] | None = None
    _ports_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_sparams(
        cls,
        sparams: dict[tuple[str, str], object],
        frequencies: list[float] | None = None,
    ) -> "SMatrixResult":
        """Build from the raw ``calculate_sparams`` dict (values are array-likes)."""
        data = {k: np.asarray(v, dtype=complex).reshape(-1) for k, v in sparams.items()}
        return cls(data=data, frequencies=frequencies)

    def out_ports(self) -> list[str]:
        """Output (detector) ports, in first-seen order."""
        seen: list[str] = []
        for o, _ in self.data:
            if o not in seen:
                seen.append(o)
        return seen

    def in_ports(self) -> list[str]:
        """Input (source) ports, in first-seen order."""
        seen: list[str] = []
        for _, i in self.data:
            if i not in seen:
                seen.append(i)
        return seen

    def matrix(self, freq_index: int = 0) -> np.ndarray:
        """Dense complex S-matrix ``S[out, in]`` at one frequency (missing entries are NaN)."""
        outs, ins = self.out_ports(), self.in_ports()
        S = np.full((len(outs), len(ins)), np.nan, dtype=complex)
        for (o, i), vals in self.data.items():
            S[outs.index(o), ins.index(i)] = vals[freq_index]
        return S

    def to_dict(self) -> dict:
        """JSON-friendly representation (complex stored as ``[re, im]`` pairs)."""
        return {
            "frequencies": self.frequencies,
            "out_ports": self.out_ports(),
            "in_ports": self.in_ports(),
            "entries": [
                {
                    "output": o,
                    "input": i,
                    "values": [[float(v.real), float(v.imag)] for v in np.asarray(vals)],
                }
                for (o, i), vals in self.data.items()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SMatrixResult":
        """Inverse of :meth:`to_dict`."""
        data = {
            (e["output"], e["input"]): np.array([complex(re, im) for re, im in e["values"]], dtype=complex)
            for e in d["entries"]
        }
        return cls(data=data, frequencies=d.get("frequencies"))

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize to a JSON string, optionally writing it to ``path``."""
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(s)
        return s

    @classmethod
    def load_json(cls, path: str | Path) -> "SMatrixResult":
        """Load from a JSON file written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def plot_smatrix(
    result: SMatrixResult,
    freq_index: int = 0,
    value: Literal["magnitude", "magnitude_db", "phase"] = "magnitude",
    filename: str | Path | None = None,
) -> "Figure":
    """Render the S-matrix at one frequency as an annotated heatmap table.

    Args:
        result: the :class:`SMatrixResult`.
        freq_index: which frequency slice to show.
        value: ``"magnitude"`` (|S|), ``"magnitude_db"`` (20·log10|S|), or ``"phase"`` (deg).
        filename: if given, save (300 dpi) and close.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    S = result.matrix(freq_index)
    outs, ins = result.out_ports(), result.in_ports()

    if value == "magnitude":
        Z, label, fmt, cmap = np.abs(S), "|S|", "{:.3f}", "viridis"
    elif value == "magnitude_db":
        Z, label, fmt, cmap = 20.0 * np.log10(np.abs(S) + 1e-12), "|S| (dB)", "{:.1f}", "viridis"
    else:
        Z, label, fmt, cmap = np.angle(S, deg=True), "phase (deg)", "{:.0f}", "twilight"

    fig, ax = plt.subplots(figsize=(1.6 * len(ins) + 2, 1.2 * len(outs) + 2))
    im = ax.imshow(Z, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ins)), ins, rotation=30, ha="right")
    ax.set_yticks(range(len(outs)), outs)
    ax.set_xlabel("input port")
    ax.set_ylabel("output port")
    title = "S-matrix" if result.frequencies is None else f"S-matrix @ {result.frequencies[freq_index]:.3e} Hz"
    ax.set_title(title)
    for r in range(len(outs)):
        for cidx in range(len(ins)):
            if np.isfinite(Z[r, cidx]):
                ax.text(cidx, r, fmt.format(Z[r, cidx]), ha="center", va="center", color="w", fontsize=9)
    plt.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
    return fig
