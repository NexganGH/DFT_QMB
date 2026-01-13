# plot_magnetisation_DFT_minus_TB.py
#
# ZGNR schematic magnetization map with zigzag chains:
# plots the DIFFERENCE per strand:  Δm = m_DFT - m_TB
#
# Requires zgnr_*_tb_fit.npz to contain:
#   mA_dft, mB_dft, mA_tb, mB_tb
#
# IMPORTANT: outputs are saved in the SAME FOLDER as this script, regardless of CWD.
#
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ----------------------------
# Paths (always relative to this script)
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NPZ_FILENAME = "zgnr_Ny2_M1_tb_fit.npz"  # <-- change to Ny8 file as needed
NPZ_PATH = os.path.join(SCRIPT_DIR, NPZ_FILENAME)

OUT_PNG = os.path.join(SCRIPT_DIR, "diff_mag_lattice_DFT_minus_TB.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "diff_mag_lattice_DFT_minus_TB.pdf")

# ----------------------------
# Geometry / style config
# ----------------------------
N_REPEAT  = 6     # AB pairs per strand (controls length)
DX_STEP   = 0.9

DY_STRAND = 1.6
Y_ZIG     = 0.35

MARKER_SIZE = 260
EDGE_LW     = 0.9
BOND_LW     = 1.1
BOND_ALPHA  = 0.85

CMAP = "RdBu_r"

# You used successfully:
#   starts_with_A = (m % 2 == 1)
STARTS_WITH_A_PARITY_IS_1 = True  # True -> (m%2==1); False -> (m%2==0)


# ----------------------------
# Helpers
# ----------------------------
def load_mA_mB_from_tbfit(npz_path, which="tb"):
    """
    Load magnetization per strand for sublattices A/B from the tb_fit npz.
    Expected keys:
      - TB:  mA_tb, mB_tb
      - DFT: mA_dft, mB_dft
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    which = which.lower().strip()
    if which == "tb":
        kA, kB = "mA_tb", "mB_tb"
    elif which == "dft":
        kA, kB = "mA_dft", "mB_dft"
    else:
        raise ValueError("which must be 'tb' or 'dft'")

    if kA not in keys or kB not in keys:
        raise KeyError(
            f"Cannot find required keys for which='{which}': need '{kA}', '{kB}'.\n"
            f"Available keys: {sorted(keys)}"
        )

    mA = np.asarray(d[kA], dtype=float).ravel()
    mB = np.asarray(d[kB], dtype=float).ravel()
    if mA.shape != mB.shape:
        raise ValueError(f"{kA} and {kB} have different shapes: {mA.shape} vs {mB.shape}")

    return mA, mB


def build_zigzag_strands(mA, mB, n_repeat, dx_step, dy_strand, y_zig, starts_with_A_parity_is_1=True):
    """
    Build zigzag polyline chains (one per strand) with alternating A/B sites.
    """
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


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(
            f"Cannot find TB fit file:\n  {NPZ_PATH}\n"
            f"(script dir: {SCRIPT_DIR})"
        )

    # --- Load DFT and TB magnetization per strand (A/B) ---
    mA_dft, mB_dft = load_mA_mB_from_tbfit(NPZ_PATH, which="dft")
    mA_tb,  mB_tb  = load_mA_mB_from_tbfit(NPZ_PATH, which="tb")

    if mA_dft.shape != mA_tb.shape:
        raise ValueError(f"Ny mismatch between DFT and TB: {mA_dft.shape} vs {mA_tb.shape}")

    # --- Difference ---
    dA = mA_dft - mA_tb
    dB = mB_dft - mB_tb

    # --- Build schematic coordinates using Δm as the color data ---
    XA, YA, CA, XB, YB, CB, bonds = build_zigzag_strands(
        dA, dB,
        n_repeat=N_REPEAT,
        dx_step=DX_STEP,
        dy_strand=DY_STRAND,
        y_zig=Y_ZIG,
        starts_with_A_parity_is_1=STARTS_WITH_A_PARITY_IS_1
    )

    # --- Symmetric color normalization around 0 ---
    all_m = np.concatenate([CA, CB])
    maxabs = float(np.max(np.abs(all_m))) if all_m.size else 1.0
    if maxabs == 0.0:
        maxabs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-maxabs, vmax=+maxabs)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)

    for (x1, y1), (x2, y2) in bonds:
        ax.plot([x1, x2], [y1, y2], color="black", lw=BOND_LW, alpha=BOND_ALPHA, zorder=1)

    scA = ax.scatter(
        XA, YA, c=CA, cmap=CMAP, norm=norm,
        s=MARKER_SIZE, marker="o",
        edgecolors="black", linewidths=EDGE_LW,
        zorder=3
    )
    ax.scatter(
        XB, YB, c=CB, cmap=CMAP, norm=norm,
        s=MARKER_SIZE, marker="s",
        edgecolors="black", linewidths=EDGE_LW,
        zorder=3
    )

    cbar = fig.colorbar(scA, ax=ax, pad=0.02)
    cbar.set_label("Δ Magnetization per site  (DFT − TB)  (μB)", fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.18)
    ax.set_aspect("equal", adjustable="box")

    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='0.7', markeredgecolor='0.2',
               markersize=10, label='A'),
        Line2D([0], [0], marker='s', color='none',
               markerfacecolor='0.7', markeredgecolor='0.2',
               markersize=10, label='B')
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="upper right", labelspacing=0.4)

    xx = np.concatenate([XA, XB])
    yy = np.concatenate([YA, YB])
    ax.set_xlim(xx.min() - 0.3, xx.max() + 0.3)
    ax.set_ylim(yy.min() - 0.6, yy.max() + 0.6)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)

    plt.show()

    print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")


if __name__ == "__main__":
    main()
