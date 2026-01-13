# plot_pretty_magn_profile_all.py
#
# Pretty magnetisation profiles for all Ny cases (one run).
# - No title
# - x-label: "strand index" (no m)
# - y-label: "magnetization (μB)" (no m_m)
# - TB: single blue color
# - DFT: single soft red, slightly transparent (less aggressive than green)
# - A: circles, B: squares
# - Legend shows TB/DFT and A/B markers (fixed position)
# - Dashed horizontal line at magnetization = 0
# - Grid
# - N, t, U in a small box (fixed position, GIF-safe)
# - Saves into:
#     NyX/<SUBFOLDER>/magn_profile/magn_profile_pretty.png
#
# Run from anywhere:
#   /home/feddex/gpaw-env/bin/python /path/to/plot_pretty_magn_profile_all.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# SETTINGS
# =========================
SUBFOLDER = "central band selection, fitting and band magnetisation graphs"
TB_FIT_PATTERN = "zgnr_*_tb_fit.npz"

OUT_DIRNAME = "magn_profile"
OUT_NAME = "magn_profile_pretty.png"

# --- COLORS / STYLES ---
COLOR_TB = "tab:blue"

# Soft muted red (publication-friendly) + a bit of transparency
COLOR_DFT = "#c44e52"
DFT_ALPHA = 0.85

LS_TB = "-"
LS_DFT = "--"

MARK_A = "o"   # circles
MARK_B = "s"   # squares

# --- FIGURE STYLE ---
FIGSIZE = (7.6, 4.4)
DPI = 320
LW = 2.6
MS = 8.5

FONTSIZE = 13
LEG_FONTSIZE = 12
BOX_FONTSIZE = 12

LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.97
BOTTOM_MARGIN = 0.16
TOP_MARGIN = 0.95

# Fixed info box position INSIDE axes (same across all plots for GIF)
BOX_AX_X = 0.52   # center-ish
BOX_AX_Y = 0.32   # lower-middle, below zero line
# =========================


def resolve_sweep_dir():
    """
    Prefer 'zgnr_sweep_results' next to this script.
    Fallback: if cwd contains zgnr_sweep_results, use it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "zgnr_sweep_results")
    if os.path.isdir(cand):
        return cand

    cwd = os.getcwd()
    cand2 = os.path.join(cwd, "zgnr_sweep_results")
    if os.path.isdir(cand2):
        return cand2

    # last fallback: maybe user is already inside it
    if any(os.path.isdir(p) for p in glob.glob(os.path.join(cwd, "Ny*"))):
        return cwd

    return None


def load_tbfit(path):
    d = np.load(path, allow_pickle=True)

    # expected keys in your tb_fit npz
    Ny = int(np.array(d["Ny"]).item()) if "Ny" in d else None
    t = float(np.array(d["t_fit"]).item()) if "t_fit" in d else None
    U = float(np.array(d["U_fit"]).item()) if "U_fit" in d else None

    mA_dft = np.array(d["mA_dft"], float).ravel()
    mB_dft = np.array(d["mB_dft"], float).ravel()
    mA_tb = np.array(d["mA_tb"], float).ravel()
    mB_tb = np.array(d["mB_tb"], float).ravel()

    return Ny, t, U, mA_dft, mB_dft, mA_tb, mB_tb


def global_ylim(series):
    vmax = max(float(np.max(np.abs(x))) for x in series)
    if vmax == 0.0:
        vmax = 1e-3
    pad = 0.12 * vmax
    return (-vmax - pad, vmax + pad)


def make_plot(Ny, t, U, mA_dft, mB_dft, mA_tb, mB_tb, ylims, outpath):
    plt.rcParams.update({"font.size": FONTSIZE, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        bottom=BOTTOM_MARGIN, top=TOP_MARGIN)

    x = np.arange(Ny)

    # dashed horizontal line at magnetization=0 (as requested)
    ax.axhline(0.0, color="0.55", lw=1.4, ls="--", zorder=0)

    # grid
    ax.grid(True, alpha=0.25)

    # TB (blue)
    ax.plot(x, mA_tb, color=COLOR_TB, lw=LW, ls=LS_TB, marker=MARK_A, ms=MS)
    ax.plot(x, mB_tb, color=COLOR_TB, lw=LW, ls=LS_TB, marker=MARK_B, ms=MS)

    # DFT (soft red, dashed, slightly transparent)
    ax.plot(x, mA_dft, color=COLOR_DFT, alpha=DFT_ALPHA,
            lw=LW, ls=LS_DFT, marker=MARK_A, ms=MS)
    ax.plot(x, mB_dft, color=COLOR_DFT, alpha=DFT_ALPHA,
            lw=LW, ls=LS_DFT, marker=MARK_B, ms=MS)

    # labels (no title)
    ax.set_xlabel("strand index")
    ax.set_ylabel("magnetization (μB)")

    ax.set_xlim(-0.5, Ny - 0.5)
    ax.set_ylim(*ylims)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend (fixed position for GIF stability)
    handles = [
        Line2D([0], [0], color=COLOR_TB, lw=LW, ls=LS_TB,
               marker=MARK_A, ms=MS, label="TB  A"),
        Line2D([0], [0], color=COLOR_TB, lw=LW, ls=LS_TB,
               marker=MARK_B, ms=MS, label="TB  B"),
        Line2D([0], [0], color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls=LS_DFT,
               marker=MARK_A, ms=MS, label="DFT A"),
        Line2D([0], [0], color=COLOR_DFT, alpha=DFT_ALPHA, lw=LW, ls=LS_DFT,
               marker=MARK_B, ms=MS, label="DFT B"),
    ]
    ax.legend(handles=handles, loc="upper right",
              frameon=True, framealpha=0.95, fontsize=LEG_FONTSIZE)

    # info box (fixed location, inside axes, where you indicated)
    t_str = f"{t:.4f}" if t is not None else "—"
    U_str = f"{U:.4f}" if U is not None else "—"
    info = (
        rf"$N = {Ny}$" + "\n" +
        rf"$t = {t_str}\ \mathrm{{eV}}$" + "\n" +
        rf"$U = {U_str}\ \mathrm{{eV}}$"
    )
    ax.text(
        BOX_AX_X, BOX_AX_Y, info,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=BOX_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95)
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def main():
    sweep = resolve_sweep_dir()
    if sweep is None:
        print("[ERROR] Could not locate zgnr_sweep_results (next to script or in cwd).")
        return

    ny_dirs = [p for p in glob.glob(os.path.join(sweep, "Ny*")) if os.path.isdir(p)]
    if not ny_dirs:
        print("[ERROR] No Ny* folders found in:", sweep)
        return

    # collect all series to set consistent y-limits (GIF-safe)
    tasks = []
    all_series = []

    # sort Ny numerically
    def ny_key(p):
        base = os.path.basename(p)
        if base.lower().startswith("ny"):
            try:
                return int(base[2:])
            except ValueError:
                return 10**9
        return 10**9

    for ny in sorted(ny_dirs, key=ny_key):
        work = os.path.join(ny, SUBFOLDER)
        if not os.path.isdir(work):
            print(f"[SKIP] {os.path.basename(ny)}: missing '{SUBFOLDER}'")
            continue

        fits = sorted(glob.glob(os.path.join(work, TB_FIT_PATTERN)))
        if not fits:
            print(f"[SKIP] {os.path.basename(ny)}: missing {TB_FIT_PATTERN}")
            continue

        try:
            Ny, t, U, mA_dft, mB_dft, mA_tb, mB_tb = load_tbfit(fits[0])
        except Exception as e:
            print(f"[FAIL] {os.path.basename(ny)}: {e}")
            continue

        if Ny is None:
            print(f"[FAIL] {os.path.basename(ny)}: Ny missing in npz")
            continue

        tasks.append((ny, Ny, t, U, mA_dft, mB_dft, mA_tb, mB_tb))
        all_series += [mA_dft, mB_dft, mA_tb, mB_tb]

    if not tasks:
        print("[ERROR] No valid Ny cases to plot.")
        return

    ylims = global_ylim(all_series)

    for ny, Ny, t, U, mA_dft, mB_dft, mA_tb, mB_tb in tasks:
        out = os.path.join(ny, SUBFOLDER, OUT_DIRNAME, OUT_NAME)
        make_plot(Ny, t, U, mA_dft, mB_dft, mA_tb, mB_tb, ylims, out)
        print(f"[OK] {os.path.basename(ny)} -> {out}")

    print("DONE.")


if __name__ == "__main__":
    main()
