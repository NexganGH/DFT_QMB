# bandplot.py

import numpy as np
import matplotlib.pyplot as plt


def compute_kpath_distance(kpts):
    """Return cumulative distance along k-path."""
    dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))


def plot_bands(kpts,
               bands_dict,
               title="Band Structure",
               xlabel="k-path",
               ylabel="Energy (eV)",
               EF_zero=True,
               figsize=(8, 5)):
    """
    Flexible band plotter.

    Parameters
    ----------
    kpts : (Nk, 3) array
        k-point coordinates along the path.
    bands_dict : dict
        Dictionary mapping labels -> dict with:
           {
               "E": (Nk, nb) array of energies,
               "color": str,
               "ls": str,
               "lw": float
           }
    title : str
    figsize : tuple

    Example of bands_dict:
        {
            "DFT π":      {"E": E_pi,    "color":"k","ls":"-","lw":2},
            "TB initial":{"E": E0,      "color":"b","ls":"-","lw":1.5},
            "TB fit":    {"E": Efit,    "color":"r","ls":"--","lw":1.2}
        }
    """

    x = compute_kpath_distance(kpts)

    plt.figure(figsize=figsize)

    for label, opts in bands_dict.items():
        E = opts["E"]
        color = opts.get("color", "k")
        ls = opts.get("ls", "-")
        lw = opts.get("lw", 1.5)

        nb = E.shape[1]  # number of bands
        for i in range(nb):
            plt.plot(x, E[:, i], color=color, ls=ls, lw=lw,
                     label=label if i == 0 else "")

    if EF_zero:
        plt.axhline(0, color='gray', lw=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
