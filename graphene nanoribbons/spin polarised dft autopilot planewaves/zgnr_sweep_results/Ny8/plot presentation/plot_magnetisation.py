# plot_magnetization_lattice.py
#
# ZGNR schematic magnetization map with zigzag chains:
#  - Along each strand: A-B-A-B-... (alternating along x)
#  - Between strands: starting sublattice alternates (A-start, B-start, A-start, ...)
#  - A: circles, B: squares
#  - Color = magnetization, shared symmetric colorbar centered at 0
#  - Bonds ONLY along each strand (between consecutive sites), no inter-strand bonds
#
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

NPZ_PATH = "zgnr_Ny8_M1_mag_AB_strands.npz"

N_REPEAT  = 10     # AB pairs per strand (length control)
DX_STEP   = 0.9
DY_STRAND = 1.6
Y_ZIG     = 0.35

MARKER_SIZE = 260
EDGE_LW     = 0.9
BOND_LW     = 1.1
BOND_ALPHA  = 0.85

OUT_PNG = "zgnr_mag_lattice_zigzag.png"
OUT_PDF = "zgnr_mag_lattice_zigzag.pdf"


def _find_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None

def load_magnetizations(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    kA = _find_key(d, ["A", "mA", "mag_A", "magA", "m_A", "mz_A", "mzA", "M_A"])
    kB = _find_key(d, ["B", "mB", "mag_B", "magB", "m_B", "mz_B", "mzB", "M_B"])

    if kA is None or kB is None:
        keys = list(d.keys())
        one_d = [k for k in keys if np.ndim(d[k]) == 1]
        for i in range(len(one_d)):
            for j in range(i + 1, len(one_d)):
                a = np.asarray(d[one_d[i]])
                b = np.asarray(d[one_d[j]])
                if a.shape == b.shape:
                    kA, kB = one_d[i], one_d[j]
                    break
            if kA is not None and kB is not None:
                break

    if kA is None or kB is None:
        raise KeyError(f"Could not find A/B arrays in {npz_path}. Keys: {list(d.keys())}")

    mA = np.asarray(d[kA], dtype=float).ravel()
    mB = np.asarray(d[kB], dtype=float).ravel()
    if mA.shape != mB.shape:
        raise ValueError(f"A and B arrays shapes differ: {mA.shape} vs {mB.shape}")
    return mA, mB


def build_zigzag_strands(mA, mB, n_repeat, dx_step, dy_strand, y_zig):
    """
    Sites j = 0..(2*n_repeat) along x.

    KEY POINT:
      - On even strands (m even): j even -> A, j odd -> B   (starts with A)
      - On odd  strands (m odd):  j even -> B, j odd -> A   (starts with B)

    Zigzag geometry:
      y = base_y + phase * s(j) * y_zig
      s(j)=+1 for even j, -1 for odd j
      phase alternates with m to make the "up/down" zigzag alternate between strands.
    """
    Ny = len(mA)

    XA, YA, CA = [], [], []
    XB, YB, CB = [], [], []
    bonds = []

    n_sites = 2 * n_repeat + 1

    for m in range(Ny):
        base_y = m * dy_strand

        # purely geometric flip so neighboring strands look like your sketch
        phase = +1 if (m % 2 == 0) else -1

        # alternation of starting sublattice between strands (this is what you asked)
        starts_with_A = (m % 2 == 1)

        strand_xy = []

        for j in range(n_sites):
            x = j * dx_step
            s = +1 if (j % 2 == 0) else -1
            y = base_y + phase * s * y_zig

            # decide whether this site is A or B
            j_even = (j % 2 == 0)
            is_A = (j_even if starts_with_A else (not j_even))

            if is_A:
                XA.append(x); YA.append(y); CA.append(mA[m])
            else:
                XB.append(x); YB.append(y); CB.append(mB[m])

            strand_xy.append((x, y))

        # bonds only along the strand
        for j in range(n_sites - 1):
            bonds.append((strand_xy[j], strand_xy[j + 1]))

    return (np.array(XA), np.array(YA), np.array(CA),
            np.array(XB), np.array(YB), np.array(CB),
            bonds)


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Cannot find {NPZ_PATH} (cwd: {os.getcwd()})")

    mA, mB = load_magnetizations(NPZ_PATH)

    XA, YA, CA, XB, YB, CB, bonds = build_zigzag_strands(
        mA, mB, N_REPEAT, DX_STEP, DY_STRAND, Y_ZIG
    )

    all_m = np.concatenate([CA, CB])
    maxabs = float(np.max(np.abs(all_m))) if all_m.size else 1.0
    if maxabs == 0.0:
        maxabs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-maxabs, vmax=+maxabs)

    fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)

    # bonds (under markers)
    for (x1, y1), (x2, y2) in bonds:
        ax.plot([x1, x2], [y1, y2], color="black", lw=BOND_LW, alpha=BOND_ALPHA, zorder=1)

    scA = ax.scatter(
        XA, YA, c=CA, cmap="RdBu_r", norm=norm,
        s=MARKER_SIZE, marker="o", edgecolors="black", linewidths=EDGE_LW,
        zorder=3, label="A"
    )
    ax.scatter(
        XB, YB, c=CB, cmap="RdBu_r", norm=norm,
        s=MARKER_SIZE, marker="s", edgecolors="black", linewidths=EDGE_LW,
        zorder=3, label="B"
    )

    cbar = fig.colorbar(scA, ax=ax, pad=0.02)
    cbar.set_label("Magnetization per site (μB)", fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=13)

    #ax.set_xlabel("Ribbon direction (schematic)")
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(True, alpha=0.18)
    ax.set_aspect("equal", adjustable="box")
    #ax.legend(loc="upper right", frameon=True)
    #ax.legend(
    #     frameon=True,
    #    loc="upper right",
    #    handletextpad=0.6,
    #    labelspacing=1  # <-- this is the key line
    #)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='0.7', markeredgecolor='0.2',
               markersize=10, label='A'),
        Line2D([0], [0], marker='s', color='none',
               markerfacecolor='0.7', markeredgecolor='0.2',
               markersize=10, label='B')
    ]

    ax.legend(
        handles=legend_elements,
        frameon=True,
        loc="upper right",
        labelspacing=0.4
    )

    xx = np.concatenate([XA, XB])
    yy = np.concatenate([YA, YB])
    ax.set_xlim(xx.min() - 0.8, xx.max() + 0.8)
    ax.set_ylim(yy.min() - 0.9, yy.max() + 0.9)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.show()

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
