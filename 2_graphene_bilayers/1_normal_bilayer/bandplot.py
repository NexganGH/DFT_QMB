# bandplot.py

import numpy as np
import matplotlib.pyplot as plt
from common.mpl_style import set_mpl_style
# This file is for plotting.

def compute_kpath_distance(kpts):
    """Return cumulative distance along k-path."""
    dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))


def plot_bands(
    kpts,
    bands_dict,
    *,
    special_points=None,
    title="Band Structure",
    xlabel="k-path",
    ylabel="Energy (eV)",
    EF_zero=True,
    figsize=(8, 5),
    outfile="../data/bands.pdf"
):
    """
    Flexible band plotter with optional high-symmetry labels.

    Parameters
    ----------
    kpts : (Nk, 3) array
        k-point coordinates along the path (cartesian).
    bands_dict : dict
        Mapping label -> plotting options.
    special_points : dict or None
        {
            "indices": list[int],
            "labels":  list[str]
        }
    """

    set_mpl_style()

    x = compute_kpath_distance(kpts)

    fig, ax = plt.subplots(figsize=figsize)

    # ---- plot bands ----
    for label, opts in bands_dict.items():
        E = opts["E"]
        color = opts.get("color", "k")
        ls = opts.get("ls", "-")
        lw = opts.get("lw", 1.5)

        nb = E.shape[1]
        for i in range(nb):
            ax.plot(
                x,
                E[:, i],
                color=color,
                ls=ls,
                lw=lw,
                label=label if i == 0 else None
            )

    # ---- Fermi level ----
    if EF_zero:
        ax.axhline(0, color="gray", lw=0.6)

    # ---- high-symmetry points ----
    if special_points is not None:
        for label, i in special_points.items():
            xi = x[i]

            # vertical line at high-symmetry point
            ax.axvline(xi, color="gray", lw=0.8, alpha=0.7)

            # label slightly below x-axis
            ax.text(
                xi,
                -0.08,  # relative to axis
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=plt.rcParams["axes.labelsize"],
            )

    # ---- cosmetics ----
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=500)
    plt.show()
