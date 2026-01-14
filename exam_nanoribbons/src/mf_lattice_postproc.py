# src/mf_lattice_postproc.py
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


# -------------------------------
# helpers
# -------------------------------
def safe_get(npz, keys):
    for k in keys:
        if k in npz:
            return npz[k]
    return None


def infer_run_dir(run_npz_path: str) -> str:
    """
    Given .../data/run_summary.npz -> return parent run dir .../
    Works even if path separators differ.
    """
    p = run_npz_path.replace("\\", "/")
    if "/data/" in p:
        return p.split("/data/")[0]
    return os.path.dirname(run_npz_path)


def find_run_summaries(base_dir: str, basename: str = "run_summary.npz"):
    files = glob.glob(os.path.join(base_dir, "**", basename), recursive=True)
    files.sort()
    return files


def load_run_info(run_npz_path: str):
    d = np.load(run_npz_path, allow_pickle=True)

    N_val = safe_get(d, ["N", "Ny"])
    if N_val is None:
        return None

    N = int(np.array(N_val).item())
    mA = np.array(d["mA"], dtype=float).ravel()
    mB = np.array(d["mB"], dtype=float).ravel()
    run_dir = infer_run_dir(run_npz_path)
    return N, mA, mB, run_dir


# -----------------------------------------------------------
# Geometry builder
# -----------------------------------------------------------
def build_zigzag_strands(
    mA, mB,
    n_repeat=12,
    dx_step=0.90,
    y_zig=0.35,
    dy_strand=1.60,
    phase_shift=True,
):
    """
    Build schematic set of N strands.
    Bonds only along each strand.
    HARD RULE:
      - B sites (squares) ALWAYS above A (circles) in each strand.
    """
    N = len(mA)
    n_sites = 2 * n_repeat + 1  # A-B-A-B-...-A

    XA, YA, CA = [], [], []
    XB, YB, CB = [], [], []
    bonds = []

    for s in range(N):
        # put strand s=0 at top (visual)
        y_base = (N - 1 - s) * dy_strand

        # stagger in x to mimic appearance
        x_phase = dx_step if (phase_shift and (s % 2 == 1)) else 0.0

        strand_xy = []
        for j in range(n_sites):
            x = x_phase + j * dx_step

            if (j % 2) == 0:
                # A site (circle) lower
                y = y_base - y_zig
                XA.append(x); YA.append(y); CA.append(mA[s])
            else:
                # B site (square) upper
                y = y_base + y_zig
                XB.append(x); YB.append(y); CB.append(mB[s])

            strand_xy.append((x, y))

        # bonds along strand only
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
def plot_lattice(
    mA, mB, N,
    out_png,
    *,
    vabs,
    box_mode="auto",
    ref_box=None,
    flip_width_order=True,
    n_repeat=12,
    dx_step=0.90,
    y_zig=0.35,
    dy_strand=1.60,
    figsize=(12.0, 5.8),
    dpi=250,
    marker_size=260,
    edge_lw=1.2,
    bond_lw=1.2,
    bond_alpha=0.90,
    cbar_fraction=0.045,
    cbar_pad=0.03,
):
    """
    box_mode:
      - "auto": tight box per N
      - "fixed": use ref_box = (xmin,xmax,ymin,ymax) for GIF stability
    """
    mA = np.array(mA, dtype=float).ravel()
    mB = np.array(mB, dtype=float).ravel()

    if flip_width_order:
        mA = mA[::-1]
        mB = mB[::-1]

    XA, YA, CA, XB, YB, CB, bonds = build_zigzag_strands(
        mA, mB,
        n_repeat=n_repeat,
        dx_step=dx_step,
        y_zig=y_zig,
        dy_strand=dy_strand,
        phase_shift=True,
    )

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap = "RdBu_r"

    fig, ax = plt.subplots(figsize=figsize)

    # bonds
    for (x1, y1), (x2, y2) in bonds:
        ax.plot([x1, x2], [y1, y2], color="black", lw=bond_lw, alpha=bond_alpha, zorder=1)

    # B first (colorbar handle)
    scB = ax.scatter(
        XB, YB, c=CB, cmap=cmap, norm=norm,
        s=marker_size, marker="s",
        edgecolors="black", linewidths=edge_lw, zorder=3,
    )
    ax.scatter(
        XA, YA, c=CA, cmap=cmap, norm=norm,
        s=marker_size, marker="o",
        edgecolors="black", linewidths=edge_lw, zorder=3,
    )

    # framing
    if box_mode == "fixed" and ref_box is not None:
        xmin, xmax, ymin, ymax = ref_box
    else:
        xmin, xmax, ymin, ymax = compute_bounds(XA, YA, XB, YB, pad_x=0.8, pad_y=0.9)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)

    # N label
    ax.text(
        0.02, 0.98, f"$N={N}$",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
    )

    # marker-only legend
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="0.7", markeredgecolor="black", label="A", markersize=10),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="0.7", markeredgecolor="black", label="B", markersize=10),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)

    cbar = fig.colorbar(scB, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.set_label("Magnetization per site (μB)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# -----------------------------------------------------------
# Main driver for post-processing
# -----------------------------------------------------------
def run_lattice_postprocessing(
    *,
    base_dir: str,
    n_min: int,
    n_max: int,
    lattice_max_n: int,
    npz_basename: str = "run_summary.npz",
    out_auto_name: str = "mag_lattice_auto.png",
    out_fixed_name: str = "mag_lattice_fixed.png",
    all_dir_auto: str = "_ALL/lattice_mag_auto",
    all_dir_fixed: str = "_ALL/lattice_mag_fixed",
    flip_width_order: bool = True,
    n_repeat: int = 12,
    dx_step: float = 0.90,
    y_zig: float = 0.35,
    dy_strand: float = 1.60,
    figsize=(12.0, 5.8),
    dpi: int = 250,
    fix_pad_x: float = 0.80,
    fix_pad_y: float = 0.90,
):
    """
    Scans base_dir recursively for run_summary.npz, and for each run:
      - if n_min <= N <= n_max and N <= lattice_max_n:
          writes lattice plots into each run_dir/magnetization/
      - copies frames into base_dir/_ALL/lattice_mag_*
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"BASE_DIR not found: {base_dir}")

    run_summaries = find_run_summaries(base_dir, basename=npz_basename)
    if not run_summaries:
        print(f"[ERROR] No {npz_basename} found under: {base_dir}")
        return

    # collect eligible runs
    runs = []
    available_N = set()

    for f in run_summaries:
        info = load_run_info(f)
        if info is None:
            continue
        N, mA, mB, run_dir = info
        available_N.add(N)

        if (n_min <= N <= n_max) and (N <= lattice_max_n):
            runs.append((N, mA, mB, run_dir, f))

    if not runs:
        print(f"[ERROR] No runs matched N in [{n_min},{n_max}] with N <= {lattice_max_n}.")
        print("[INFO] Available N found:", sorted(available_N))
        return

    runs.sort(key=lambda x: x[0])

    # global color scale for consistency
    vabs = 0.0
    for (N, mA, mB, run_dir, f) in runs:
        mA2 = mA[::-1] if flip_width_order else mA
        mB2 = mB[::-1] if flip_width_order else mB
        vabs = max(vabs, float(np.max(np.abs(mA2))), float(np.max(np.abs(mB2))))
    vabs = max(vabs, 1e-12)

    # fixed reference box from the largest N among eligible runs
    N_ref, mA_ref, mB_ref, run_dir_ref, f_ref = runs[-1]
    mA_ref2 = mA_ref[::-1] if flip_width_order else mA_ref
    mB_ref2 = mB_ref[::-1] if flip_width_order else mB_ref

    XA, YA, CA, XB, YB, CB, bonds = build_zigzag_strands(
        mA_ref2, mB_ref2,
        n_repeat=n_repeat, dx_step=dx_step, y_zig=y_zig, dy_strand=dy_strand
    )
    ref_box = compute_bounds(XA, YA, XB, YB, pad_x=fix_pad_x, pad_y=fix_pad_y)

    # prepare _ALL dirs
    all_auto_dir = os.path.join(base_dir, all_dir_auto)
    all_fixed_dir = os.path.join(base_dir, all_dir_fixed)
    os.makedirs(all_auto_dir, exist_ok=True)
    os.makedirs(all_fixed_dir, exist_ok=True)

    # generate plots per run
    for (N, mA, mB, run_dir, f) in runs:
        mag_dir = os.path.join(run_dir, "magnetization")
        os.makedirs(mag_dir, exist_ok=True)

        out_auto = os.path.join(mag_dir, out_auto_name)
        out_fixed = os.path.join(mag_dir, out_fixed_name)

        plot_lattice(
            mA, mB, N, out_auto,
            vabs=vabs, box_mode="auto", ref_box=None,
            flip_width_order=flip_width_order,
            n_repeat=n_repeat, dx_step=dx_step, y_zig=y_zig, dy_strand=dy_strand,
            figsize=figsize, dpi=dpi,
        )
        plot_lattice(
            mA, mB, N, out_fixed,
            vabs=vabs, box_mode="fixed", ref_box=ref_box,
            flip_width_order=flip_width_order,
            n_repeat=n_repeat, dx_step=dx_step, y_zig=y_zig, dy_strand=dy_strand,
            figsize=figsize, dpi=dpi,
        )

        shutil.copy2(out_auto, os.path.join(all_auto_dir, f"mag_lattice_auto_N{N:02d}.png"))
        shutil.copy2(out_fixed, os.path.join(all_fixed_dir, f"mag_lattice_fixed_N{N:02d}.png"))

        print("[SAVED]", out_auto)
        print("[SAVED]", out_fixed)

    print("\nDONE lattice postprocessing.")
    print("Frames collected in:")
    print("  AUTO :", all_auto_dir)
    print("  FIXED:", all_fixed_dir)
