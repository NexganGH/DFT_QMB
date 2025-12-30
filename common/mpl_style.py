# plots/mpl_style.py

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_mpl_style(
    *,
    fontsize: int = 17,
    linewidth: float = 1.8,
    use_latex: bool = True
) -> None:
    """
    Apply a global Matplotlib style for publication-quality figures.

    Call this ONCE before creating any plots.

    Parameters
    ----------
    fontsize : int
        Base font size for labels and ticks.
    linewidth : float
        Default line width.
    use_latex : bool
        Whether to enable LaTeX rendering.
    """

    mpl.rcParams.update({

        # -------- Figure --------
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 150,
        "figure.facecolor": "white",

        # -------- Axes --------
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.linewidth": 1.2,

        # -------- Text --------
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",

        # -------- Ticks --------
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,

        # -------- Lines --------
        "lines.linewidth": linewidth,

        # -------- Legend --------
        "legend.fontsize": fontsize - 1,
        "legend.frameon": False,

        # -------- Grid --------
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,

        # -------- Save --------
        "savefig.format": "pdf",
        "savefig.dpi": 500,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",

        # --- Enable LaTeX ---
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

