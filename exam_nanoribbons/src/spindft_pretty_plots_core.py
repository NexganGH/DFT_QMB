# src/spindft_pretty_plots_core.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm


# ----------------------------
# Pretty bands (from *_bands_pi.npz produced by compute_bands_pi)
# ----------------------------
def _to_0_1_axis(k):
    k = np.asarray(k, float).ravel()
    kmin, kmax = float(k.min()), float(k.max())
    return np.zeros_like(k) if np.isclose(kmax, kmin) else (k - kmin) / (kmax - kmin)

def _central_band_curves(Eshift):
    # Eshift: (Nk, nb)
    Nk, nb = Eshift.shape
    val = np.full(Nk, np.nan, float)
    con = np.full(Nk, np.nan, float)
    for i in range(Nk):
        Ek = Eshift[i]
        below = np.where(Ek <= 0.0)[0]
        above = np.where(Ek >  0.0)[0]
        if below.size:
            val[i] = Ek[below[np.argmax(Ek[below])]]
        if above.size:
            con[i] = Ek[above[np.argmin(Ek[above])]]
    return val, con

def _band_side_color(y, col_val, col_con):
    frac_below = np.mean(np.asarray(y) <= 0.0)
    return col_val if frac_below >= 0.5 else col_con

def pretty_bands_from_npz(
    bands_npz_path: str,
    out_colored_png: str,
    out_gray_png: str,
    Ny_label: int | None = None,
    figsize=(4.0, 6.0),
    dpi=300,
):
    d = np.load(bands_npz_path, allow_pickle=True)

    # k axis
    if "k_plot" in d:
        k = np.asarray(d["k_plot"], float).ravel()
    elif "k_dimless" in d:
        k = np.asarray(d["k_dimless"], float).ravel()
    elif "k_frac" in d:
        k = np.asarray(d["k_frac"], float).ravel()
    else:
        raise KeyError(f"No k array in {bands_npz_path}. Keys={list(d.keys())}")

    # energies (prefer E_rel)
    if "E_rel" in d:
        Eraw = np.asarray(d["E_rel"], float)  # (nspins, Nk, nb) or (Nk, nb)
        mu = 0.0
    elif "energies" in d:
        Eraw = np.asarray(d["energies"], float)
        mu = float(np.array(d["efermi"]).item()) if "efermi" in d else 0.0
    else:
        raise KeyError(f"No E_rel/energies in {bands_npz_path}. Keys={list(d.keys())}")

    Nk = len(k)

    # normalize E to (Nk, nb)
    if Eraw.ndim == 2:
        E = Eraw if Eraw.shape[0] == Nk else Eraw.T
    elif Eraw.ndim == 3:
        # average spins -> (Nk, nb)
        if Eraw.shape[1] == Nk:      # (nspins, Nk, nb)
            E = np.mean(Eraw, axis=0)
        elif Eraw.shape[0] == Nk:    # (Nk, nb, nspins)
            E = np.mean(Eraw, axis=-1)
        else:
            raise ValueError(f"Bad Eraw shape {Eraw.shape} vs Nk={Nk}")
    else:
        raise ValueError(f"Bad Eraw ndim={Eraw.ndim}")

    Eshift = E - mu
    x = _to_0_1_axis(k)
    val_c, con_c = _central_band_curves(Eshift)

    # style
    COL_VAL = "#2E5EAA"
    COL_CON = "#D89C2B"
    COL_CEN = "#C23B3B"
    GRAY_OTHER = "0.75"
    LW_OTHER = 1.2
    LW_CENT = 3.2

    def common_axes(ax):
        ax.set_xlabel(r"$ka/\pi$")
        ax.set_ylabel(r"$E - E_f$ (eV)")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([r"$\Gamma$", r"$Z$"])
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, ls="--", lw=1.2, alpha=0.7, zorder=0)

        if Ny_label is not None:
            ax.text(
                0.97, 0.93, rf"$N = {Ny_label}$",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=18,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
                zorder=10
            )

    # COLORED
    fig, ax = plt.subplots(figsize=figsize)
    for b in range(Eshift.shape[1]):
        y = Eshift[:, b]
        c = _band_side_color(y, COL_VAL, COL_CON)
        ax.plot(x, y, lw=LW_OTHER, alpha=0.95, color=c, zorder=1)
    ax.plot(x, val_c, lw=LW_CENT, color=COL_CEN, zorder=3)
    ax.plot(x, con_c, lw=LW_CENT, color=COL_CEN, zorder=3)
    common_axes(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_colored_png), exist_ok=True)
    fig.savefig(out_colored_png, dpi=dpi)
    plt.close(fig)

    # GRAY
    fig, ax = plt.subplots(figsize=figsize)
    for b in range(Eshift.shape[1]):
        ax.plot(x, Eshift[:, b], lw=LW_OTHER, alpha=1.0, color=GRAY_OTHER, zorder=1)
    ax.plot(x, val_c, lw=LW_CENT, color=COL_CEN, zorder=3)
    ax.plot(x, con_c, lw=LW_CENT, color=COL_CEN, zorder=3)
    common_axes(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_gray_png), exist_ok=True)
    fig.savefig(out_gray_png, dpi=dpi)
    plt.close(fig)


# ----------------------------
# Pretty magnetisation profile (from tb_fit.npz)
# ----------------------------
def pretty_magnetization_profile_from_tbfit(
    tbfit_npz_path: str,
    out_png: str,
    Ny_label: int | None = None,
    figsize=(7.6, 4.4),
    dpi=320,
):
    d = np.load(tbfit_npz_path, allow_pickle=True)
    # Expect these keys (make sure your tbfit writer uses them)
    mA_dft = np.asarray(d["mA_dft"], float).ravel()
    mB_dft = np.asarray(d["mB_dft"], float).ravel()
    mA_tb  = np.asarray(d["mA_tb"],  float).ravel()
    mB_tb  = np.asarray(d["mB_tb"],  float).ravel()

    t = float(np.array(d["t_fit"]).item()) if "t_fit" in d else None
    U = float(np.array(d["U_fit"]).item()) if "U_fit" in d else None
    Ny = int(np.array(d["Ny"]).item()) if "Ny" in d else len(mA_tb)
    if Ny_label is None:
        Ny_label = Ny

    # y-lims symmetric
    all_series = [mA_dft, mB_dft, mA_tb, mB_tb]
    vmax = max(float(np.max(np.abs(x))) for x in all_series) or 1e-3
    pad = 0.12 * vmax
    ylims = (-vmax - pad, vmax + pad)

    COLOR_TB = "tab:blue"
    COLOR_DFT = "#c44e52"
    DFT_ALPHA = 0.85
    LW = 2.6
    MS = 8.5

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.95)

    x = np.arange(Ny)
    ax.axhline(0.0, color="0.55", lw=1.4, ls="--", zorder=0)
    ax.grid(True, alpha=0.25)

    ax.plot(x, mA_tb, color=COLOR_TB, lw=LW, ls="-",  marker="o", ms=MS)
    ax.plot(x, mB_tb, color=COLOR_TB, lw=LW, ls="-",  marker="s", ms=MS)

    ax.plot(x, mA_dft, color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls="--", marker="o", ms=MS)
    ax.plot(x, mB_dft, color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls="--", marker="s", ms=MS)

    ax.set_xlabel("strand index")
    ax.set_ylabel("magnetization (μB)")
    ax.set_xlim(-0.5, Ny - 0.5)
    ax.set_ylim(*ylims)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0],[0], color=COLOR_TB,  lw=LW, ls="-",  marker="o", ms=MS, label="TB  A"),
        Line2D([0],[0], color=COLOR_TB,  lw=LW, ls="-",  marker="s", ms=MS, label="TB  B"),
        Line2D([0],[0], color=COLOR_DFT, lw=LW, ls="--", marker="o", ms=MS, alpha=DFT_ALPHA, label="DFT A"),
        Line2D([0],[0], color=COLOR_DFT, lw=LW, ls="--", marker="s", ms=MS, alpha=DFT_ALPHA, label="DFT B"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=12)

    t_str = f"{t:.4f}" if t is not None else "—"
    U_str = f"{U:.4f}" if U is not None else "—"
    info = rf"$N = {Ny_label}$" + "\n" + rf"$t = {t_str}\ \mathrm{{eV}}$" + "\n" + rf"$U = {U_str}\ \mathrm{{eV}}$"
    ax.text(
        0.52, 0.32, info,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95),
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ----------------------------
# Zigzag lattice plots (TB or Δ=DFT-TB) from tb_fit.npz
# ----------------------------
def _build_zigzag_strands(mA, mB, n_repeat=6, dx_step=0.9, dy_strand=1.6, y_zig=0.35,
                          starts_with_A_parity_is_1=True):
    Ny = len(mA)
    XA, YA, CA = [], [], []
    XB, YB, CB = [], [], []
    bonds = []
    n_sites = 2 * n_repeat + 1

    for m in range(Ny):
        base_y = m * dy_strand
        phase = +1 if (m % 2 == 0) else -1
        starts_with_A = (m % 2 == 1) if starts_with_A_parity_is_1 else (m % 2 == 0)

        strand_xy = []
        for j in range(n_sites):
            x = j * dx_step
            s = +1 if (j % 2 == 0) else -1
            y = base_y + phase * s * y_zig

            j_even = (j % 2 == 0)
            is_A = (j_even if starts_with_A else (not j_even))

            if is_A:
                XA.append(x); YA.append(y); CA.append(mA[m])
            else:
                XB.append(x); YB.append(y); CB.append(mB[m])
            strand_xy.append((x, y))

        for j in range(n_sites - 1):
            bonds.append((strand_xy[j], strand_xy[j + 1]))

    return (np.array(XA), np.array(YA), np.array(CA),
            np.array(XB), np.array(YB), np.array(CB),
            bonds)

def lattice_map_from_tbfit(
    tbfit_npz_path: str,
    out_png: str,
    mode: str = "diff",   # "diff" (DFT-TB) or "tb"
    n_repeat=6,
    starts_with_A_parity_is_1=True,
    figsize=(14, 5.2),
    dpi=300,
):
    d = np.load(tbfit_npz_path, allow_pickle=True)
    mA_tb = np.asarray(d["mA_tb"], float).ravel()
    mB_tb = np.asarray(d["mB_tb"], float).ravel()

    if mode == "tb":
        mA, mB = mA_tb, mB_tb
        cbar_label = "Magnetization per site (μB)"
    elif mode == "diff":
        mA_dft = np.asarray(d["mA_dft"], float).ravel()
        mB_dft = np.asarray(d["mB_dft"], float).ravel()
        mA, mB = (mA_dft - mA_tb), (mB_dft - mB_tb)
        cbar_label = "Δ Magnetization per site  (DFT − TB)  (μB)"
    else:
        raise ValueError("mode must be 'tb' or 'diff'")

    XA, YA, CA, XB, YB, CB, bonds = _build_zigzag_strands(
        mA, mB, n_repeat=n_repeat, starts_with_A_parity_is_1=starts_with_A_parity_is_1
    )

    all_m = np.concatenate([CA, CB])
    maxabs = float(np.max(np.abs(all_m))) if all_m.size else 1.0
    if maxabs == 0.0:
        maxabs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-maxabs, vmax=+maxabs)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for (x1, y1), (x2, y2) in bonds:
        ax.plot([x1, x2], [y1, y2], color="black", lw=1.1, alpha=0.85, zorder=1)

    scA = ax.scatter(XA, YA, c=CA, cmap="RdBu_r", norm=norm,
                     s=260, marker="o", edgecolors="black", linewidths=0.9, zorder=3)
    ax.scatter(XB, YB, c=CB, cmap="RdBu_r", norm=norm,
               s=260, marker="s", edgecolors="black", linewidths=0.9, zorder=3)

    cbar = fig.colorbar(scA, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, alpha=0.18)
    ax.set_aspect("equal", adjustable="box")

    legend_elements = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor='0.7', markeredgecolor='0.2', markersize=10, label='A'),
        Line2D([0],[0], marker='s', color='none', markerfacecolor='0.7', markeredgecolor='0.2', markersize=10, label='B'),
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="upper right", labelspacing=0.4)

    xx = np.concatenate([XA, XB])
    yy = np.concatenate([YA, YB])
    ax.set_xlim(xx.min() - 0.3, xx.max() + 0.3)
    ax.set_ylim(yy.min() - 0.6, yy.max() + 0.6)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
