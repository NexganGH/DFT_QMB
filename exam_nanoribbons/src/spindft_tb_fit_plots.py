# spindft_tb_fit_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_pi_fit_pretty(k, dft_val, dft_cond, tb_val, tb_cond, Ny, t, U, out_png):
    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    ax.axhline(0.0, lw=1.2, ls="--", color="0.55", zorder=0)

    ax.plot(k, dft_val,  color="tab:red",  lw=3.2, ls="-",  label="DFT (central π)")
    ax.plot(k, dft_cond, color="tab:red",  lw=3.2, ls="-",  label="_nolegend_")
    ax.plot(k, tb_val,   color="tab:blue", lw=2.8, ls="--", label="TB fit")
    ax.plot(k, tb_cond,  color="tab:blue", lw=2.8, ls="--", label="_nolegend_")

    ax.set_xlabel(r"$ka/\pi$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=12)

    info = rf"$N={Ny}$" + "\n" + rf"$t={t:.4f}\ \mathrm{{eV}}$" + "\n" + rf"$U={U:.4f}\ \mathrm{{eV}}$"
    ax.text(
        0.84, 0.24, info,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=320)
    plt.close(fig)

def plot_magnetization_profile_pretty(mA_dft, mB_dft, mA_tb, mB_tb, Ny, t, U, out_png):
    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    x = np.arange(Ny)
    ax.axhline(0.0, color="0.55", lw=1.4, ls="--", zorder=0)
    ax.grid(True, alpha=0.25)

    COLOR_TB = "tab:blue"
    COLOR_DFT = "#c44e52"
    DFT_ALPHA = 0.85
    LW = 2.6
    MS = 8.5

    ax.plot(x, mA_tb, color=COLOR_TB, lw=LW, ls="-",  marker="o", ms=MS)
    ax.plot(x, mB_tb, color=COLOR_TB, lw=LW, ls="-",  marker="s", ms=MS)
    ax.plot(x, mA_dft, color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls="--", marker="o", ms=MS)
    ax.plot(x, mB_dft, color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls="--", marker="s", ms=MS)

    ax.set_xlabel("strand index")
    ax.set_ylabel("magnetization (μB)")
    ax.set_xlim(-0.5, Ny - 0.5)

    handles = [
        Line2D([0], [0], color=COLOR_TB, lw=LW, ls="-",  marker="o", ms=MS, label="TB  A"),
        Line2D([0], [0], color=COLOR_TB, lw=LW, ls="-",  marker="s", ms=MS, label="TB  B"),
        Line2D([0], [0], color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls="--", marker="o", ms=MS, label="DFT A"),
        Line2D([0], [0], color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls="--", marker="s", ms=MS, label="DFT B"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=12)

    info = rf"$N={Ny}$" + "\n" + rf"$t={t:.4f}\ \mathrm{{eV}}$" + "\n" + rf"$U={U:.4f}\ \mathrm{{eV}}$"
    ax.text(
        0.52, 0.32, info,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=320)
    plt.close(fig)
