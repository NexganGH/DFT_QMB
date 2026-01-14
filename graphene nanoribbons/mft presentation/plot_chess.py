# plot_chess.py
#
# ZGNR MF magnetization lattice schematic from saved run_summary.npz files.
# - Bonds only along each strand (no inter-strand bonds)
# - B (squares) ALWAYS above A (circles) on every strand
# - Optional flip of strand order so that "blue B edge" is on top and "red A edge" on bottom
# - Saves:
#     magnetization/mag_lattice_auto.png   (auto box)
#     magnetization/mag_lattice_fixed.png  (fixed box for GIF consistency)
# - Collects fixed frames in: _ALL/lattice_mag_fixed/

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ===========================================================
# USER SETTINGS
# ===========================================================
BASE_DIR = "postproc_outputs"      # or "postproc_outputs_U_sweep"
NPZ_BASENAME = "run_summary.npz"

N_MIN, N_MAX = 2, 8               # sweep limits (inclusive)

# Longitudinal repetition (controls "how wide" the drawing is)
# Increase this if you want a wider/cleaner representation.
N_REPEAT = 12                      # number of AB pairs per strand
DX_STEP  = 0.90                    # horizontal step between consecutive sites
Y_ZIG    = 0.35                    # vertical zigzag amplitude inside a strand
DY_STRAND = 1.60                   # vertical spacing between strands

# IMPORTANT: this fixes your “blue squares on top / red circles on bottom”
# If your plot looks vertically “reversed” in terms of edge colors, set True.
FLIP_WIDTH_ORDER = True

# Box modes saved
OUT_AUTO  = "mag_lattice_auto.png"
OUT_FIXED = "mag_lattice_fixed.png"

# Where to collect frames for GIFs (under BASE_DIR)
ALL_DIR_FIXED = "_ALL/lattice_mag_fixed"
ALL_DIR_AUTO  = "_ALL/lattice_mag_auto"

# Styling
FIGSIZE = (12.0, 5.8)
DPI = 250

MARKER_SIZE = 260
EDGE_LW = 1.2
BOND_LW = 1.2
BOND_ALPHA = 0.90

CBAR_FRACTION = 0.045
CBAR_PAD = 0.03

# Fixed-box padding (extra whitespace around the lattice)
FIX_PAD_X = 0.80
FIX_PAD_Y = 0.90
# ===========================================================


# -------------------------------
# Filesystem helpers
# -------------------------------
def resolve_base_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, BASE_DIR))


def find_run_summaries(base_dir):
    files = glob.glob(os.path.join(base_dir, "**", NPZ_BASENAME), recursive=True)
    files.sort()
    return files


def safe_get(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_run_info(run_npz):
    d = np.load(run_npz, allow_pickle=True)

    N_val = safe_get(d, ["N", "Ny"])
    if N_val is None:
        return None

    N = int(np.array(N_val).item())

    # run_dir is parent of ".../data/run_summary.npz" if present
    p = run_npz.replace("\\", "/")
    if "/data/" in p:
        run_dir = p.split("/data/")[0]
    else:
        run_dir = os.path.dirname(run_npz)

    return N, run_dir


def load_magnetization(run_npz):
    d = np.load(run_npz, allow_pickle=True)
    mA = np.array(d["mA"], dtype=float).ravel()
    mB = np.array(d["mB"], dtype=float).ravel()
    return mA, mB


# -----------------------------------------------------------
# Geometry builder
# -----------------------------------------------------------
def build_zigzag_strands(
    mA, mB,
    n_repeat=N_REPEAT,
    dx_step=DX_STEP,
    dy_strand=DY_STRAND,
    y_zig=Y_ZIG,
    phase_shift=True
):
    """
    Build a schematic set of N strands.
    Each strand is a 1D zigzag chain with alternating A/B along x.

    HARD RULE enforced here:
      - B sites (squares) are ALWAYS at y = base_y + y_zig
      - A sites (circles) are ALWAYS at y = base_y - y_zig

    We also optionally shift neighboring strands in x by half-step
    to mimic the staggered appearance without adding inter-strand bonds.
    """
    N = len(mA)
    n_sites = 2 * n_repeat + 1  # A-B-A-B-...-A

    XA, YA, CA = [], [], []
    XB, YB, CB = [], [], []
    bonds = []

    for s in range(N):
        # Put strand s=0 at top (visual ordering)
        # so we map s -> y using reversed indexing
        y_base = (N - 1 - s) * dy_strand

        # OLD (half-step -> causes the offset you don't want)
        # x_phase = 0.5 * dx_step if (phase_shift and (s % 2 == 1)) else 0.0

        # NEW (full-step -> aligns squares with circles of neighboring strands)
        x_phase = dx_step if (phase_shift and (s % 2 == 1)) else 0.0

        # store points for bonds
        strand_xy = []

        for j in range(n_sites):
            x = x_phase + j * dx_step

            if (j % 2) == 0:
                # A site (circle) ALWAYS lower
                y = y_base - y_zig
                XA.append(x); YA.append(y); CA.append(mA[s])
            else:
                # B site (square) ALWAYS upper
                y = y_base + y_zig
                XB.append(x); YB.append(y); CB.append(mB[s])

            strand_xy.append((x, y))

        # bonds only along the strand
        for j in range(n_sites - 1):
            bonds.append((strand_xy[j], strand_xy[j + 1]))

    return (np.array(XA), np.array(YA), np.array(CA),
            np.array(XB), np.array(YB), np.array(CB),
            bonds)


def compute_bounds(XA, YA, XB, YB, pad_x=0.0, pad_y=0.0):
    xx = np.concatenate([XA, XB]) if XA.size and XB.size else (XA if XA.size else XB)
    yy = np.concatenate([YA, YB]) if YA.size and YB.size else (YA if YA.size else YB)
    xmin, xmax = xx.min() - pad_x, xx.max() + pad_x
    ymin, ymax = yy.min() - pad_y, yy.max() + pad_y
    return xmin, xmax, ymin, ymax


# -----------------------------------------------------------
# Plotting
# -----------------------------------------------------------
def plot_lattice(mA, mB, N, out_png, vabs, box_mode="auto", ref_box=None):
    """
    box_mode:
      - "auto": axis limits fit the current lattice tightly (with small pad)
      - "fixed": use ref_box = (xmin,xmax,ymin,ymax) for consistent GIF frames
    """
    XA, YA, CA, XB, YB, CB, bonds = build_zigzag_strands(mA, mB)

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap = "RdBu_r"

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # bonds (under markers)
    for (x1, y1), (x2, y2) in bonds:
        ax.plot([x1, x2], [y1, y2], color="black", lw=BOND_LW, alpha=BOND_ALPHA, zorder=1)

    # scatter (B first so we can attach colorbar handle)
    scB = ax.scatter(
        XB, YB, c=CB, cmap=cmap, norm=norm,
        s=MARKER_SIZE, marker="s",
        edgecolors="black", linewidths=EDGE_LW, zorder=3
    )
    ax.scatter(
        XA, YA, c=CA, cmap=cmap, norm=norm,
        s=MARKER_SIZE, marker="o",
        edgecolors="black", linewidths=EDGE_LW, zorder=3
    )

    # framing & aspect
    if box_mode == "fixed" and ref_box is not None:
        xmin, xmax, ymin, ymax = ref_box
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        xmin, xmax, ymin, ymax = compute_bounds(XA, YA, XB, YB, pad_x=0.8, pad_y=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)

    # N label (top-left)
    ax.text(
        0.02, 0.98, f"$N={N}$",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95)
    )

    # legend (top-right), marker-only
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="0.7", markeredgecolor="black", label="A", markersize=10),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="0.7", markeredgecolor="black", label="B", markersize=10),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)

    # colorbar
    cbar = fig.colorbar(scB, ax=ax, fraction=CBAR_FRACTION, pad=CBAR_PAD)
    cbar.set_label("Magnetization per site (μB)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def main():
    base_dir = resolve_base_dir()
    print(f"[INFO] BASE_DIR: {base_dir}")

    if not os.path.isdir(base_dir):
        print(f"[ERROR] BASE_DIR not found: {base_dir}")
        return

    run_summaries = find_run_summaries(base_dir)
    print(f"[INFO] Found {len(run_summaries)} run_summary files (recursive).")

    if not run_summaries:
        print(f"[ERROR] No {NPZ_BASENAME} found under: {base_dir}")
        return

    runs = []
    available_N = set()

    for f in run_summaries:
        info = load_run_info(f)
        if info is None:
            continue
        N, run_dir = info
        available_N.add(N)
        if N_MIN <= N <= N_MAX:
            runs.append((N, run_dir, f))

    if not runs:
        print(f"[ERROR] No runs matched N in [{N_MIN},{N_MAX}].")
        print("[INFO] Available N found in npz:", sorted(available_N))
        return

    runs.sort(key=lambda x: x[0])

    # global color scale for GIF consistency
    vabs = 0.0
    for (N, run_dir, f) in runs:
        mA, mB = load_magnetization(f)
        if FLIP_WIDTH_ORDER:
            mA = mA[::-1]
            mB = mB[::-1]
        vabs = max(vabs, float(np.max(np.abs(mA))), float(np.max(np.abs(mB))))
    if vabs == 0.0:
        vabs = 1e-6

    # Build a fixed reference box from the *largest N* case (usually N_MAX)
    # so that smaller N leave empty space (good for GIF stability).
    N_ref, run_dir_ref, f_ref = runs[-1]
    mA_ref, mB_ref = load_magnetization(f_ref)
    if FLIP_WIDTH_ORDER:
        mA_ref = mA_ref[::-1]
        mB_ref = mB_ref[::-1]

    XA, YA, CA, XB, YB, CB, bonds = build_zigzag_strands(mA_ref, mB_ref)
    ref_box = compute_bounds(XA, YA, XB, YB, pad_x=FIX_PAD_X, pad_y=FIX_PAD_Y)

    # Prepare _ALL frame folders
    all_fixed_dir = os.path.join(base_dir, ALL_DIR_FIXED)
    all_auto_dir  = os.path.join(base_dir, ALL_DIR_AUTO)
    os.makedirs(all_fixed_dir, exist_ok=True)
    os.makedirs(all_auto_dir, exist_ok=True)

    # Generate plots
    for (N, run_dir, f) in runs:
        mA, mB = load_magnetization(f)

        # <<< KEY FIX FOR "blue squares on top, red circles on bottom" >>>
        if FLIP_WIDTH_ORDER:
            mA = mA[::-1]
            mB = mB[::-1]

        mag_dir = os.path.join(run_dir, "magnetization")
        os.makedirs(mag_dir, exist_ok=True)

        out_auto  = os.path.join(mag_dir, OUT_AUTO)
        out_fixed = os.path.join(mag_dir, OUT_FIXED)

        plot_lattice(mA, mB, N, out_auto,  vabs=vabs, box_mode="auto",  ref_box=None)
        plot_lattice(mA, mB, N, out_fixed, vabs=vabs, box_mode="fixed", ref_box=ref_box)

        # collect frames for GIF
        shutil.copy2(out_auto,  os.path.join(all_auto_dir,  f"mag_lattice_auto_N{N:02d}.png"))
        shutil.copy2(out_fixed, os.path.join(all_fixed_dir, f"mag_lattice_fixed_N{N:02d}.png"))

        print("Saved:", out_auto)
        print("Saved:", out_fixed)

    print("\nDONE.")
    print("Frames for GIF collected in:")
    print("  AUTO :", all_auto_dir)
    print("  FIXED:", all_fixed_dir)
    print("\nIf the edge colors are still inverted, keep everything the same and just toggle:")
    print("  FLIP_WIDTH_ORDER = False/True")


if __name__ == "__main__":
    main()
