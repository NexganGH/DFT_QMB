import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.build import graphene_nanoribbon
from ase.neighborlist import NeighborList, natural_cutoffs


def best_2d_axes(pos):
    var = pos.var(axis=0)
    ij = np.argsort(var)[-2:]
    return int(ij[0]), int(ij[1])


def rotate_2d(x, y, angle_deg):
    th = np.deg2rad(angle_deg)
    xr = x * np.cos(th) - y * np.sin(th)
    yr = x * np.sin(th) + y * np.cos(th)
    return xr, yr


def draw_xy_frame(ax, origin=(0.12, 0.18), length=0.10, lw=2.5):
    """
    Draw a small x-y frame in axes-fraction coordinates (never clipped),
    with a small white background patch.
    """
    ox, oy = origin

    pad = 0.02
    bg = plt.Rectangle(
        (ox - pad, oy - pad),
        length + 0.09, length + 0.09,
        transform=ax.transAxes,
        facecolor="white", edgecolor="none",
        zorder=2, alpha=0.9
    )
    ax.add_patch(bg)

    ax.annotate(
        "", xy=(ox + length, oy), xytext=(ox, oy),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=lw, color="black"),
        zorder=3
    )
    ax.text(ox + length + 0.02, oy - 0.01, "x",
            transform=ax.transAxes, fontsize=12, zorder=3)

    ax.annotate(
        "", xy=(ox, oy + length), xytext=(ox, oy),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=lw, color="black"),
        zorder=3
    )
    ax.text(ox - 0.02, oy + length + 0.01, "y",
            transform=ax.transAxes, fontsize=12, zorder=3)


def plot_gnr_pretty(
    atoms,
    title="Zig Zag",
    make_horizontal=True,
    color_C="#2B2D42",   # carbon
    color_H="#8ECAE6",   # hydrogen
    bond_color="#4A4A4A",
    atom_size_C=60,
    atom_size_H=26,
    bond_lw=1.2,
    show_xy_frame=True,
    savepath=None
):
    # Avoid PBC-image bonds in plotting
    atoms_plot = atoms.copy()
    atoms_plot.set_pbc((False, False, False))

    pos = atoms_plot.get_positions()
    ax_i, ax_j = best_2d_axes(pos)

    x = pos[:, ax_i].copy()
    y = pos[:, ax_j].copy()

    # Center in 2D
    x -= x.mean()
    y -= y.mean()

    # Rotate to make the ribbon horizontal
    if make_horizontal:
        cov = np.cov(np.vstack([x, y]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        angle = np.degrees(np.arctan2(v[1], v[0]))
        x, y = rotate_2d(x, y, -angle)
        if (y.max() - y.min()) > (x.max() - x.min()):
            x, y = rotate_2d(x, y, 90)

    fig, ax = plt.subplots(figsize=(10, 3))

    # Neighbor list for bonds
    cutoffs = natural_cutoffs(atoms_plot)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms_plot)

    # Draw bonds (skip periodic offsets explicitly)
    for i in range(len(atoms_plot)):
        neigh, offsets = nl.get_neighbors(i)
        for j, off in zip(neigh, offsets):
            if j <= i:
                continue
            if np.any(off != 0):
                continue

            rij = atoms_plot.get_distance(i, j, mic=False)
            Zi, Zj = atoms_plot.numbers[i], atoms_plot.numbers[j]

            # --- robust bond filters ---
            pair = tuple(sorted((Zi, Zj)))
            if pair == (6, 6):          # C–C
                if not (1.20 <= rij <= 1.90):
                    continue
            elif pair == (1, 6):        # C–H
                if not (0.85 <= rij <= 1.45):
                    continue
            else:
                continue

            ax.plot([x[i], x[j]], [y[i], y[j]],
                    color=bond_color, lw=bond_lw, zorder=1)

    # Draw atoms
    for i, Z in enumerate(atoms_plot.numbers):
        if Z == 6:
            ax.scatter(x[i], y[i], s=atom_size_C, c=color_C,
                       edgecolors="#111111", linewidths=0.7, zorder=4)
        elif Z == 1:
            ax.scatter(x[i], y[i], s=atom_size_H, c=color_H,
                       edgecolors="#111111", linewidths=0.6, zorder=5)

    # Legend outside, simple labels
    legend_elems = [
        Line2D([0], [0], marker='o', color='none', label='C',
               markerfacecolor=color_C, markeredgecolor="#111111", markersize=8),
        Line2D([0], [0], marker='o', color='none', label='H',
               markerfacecolor=color_H, markeredgecolor="#111111", markersize=6),
    ]
    ax.legend(handles=legend_elems, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0)

    ax.set_title(title, fontsize=16, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Tight framing (leave some margins)
    padx = 0.06 * (x.max() - x.min() + 1e-9)
    pady = 0.16 * (y.max() - y.min() + 1e-9)
    ax.set_xlim(x.min() - padx, x.max() + padx)
    ax.set_ylim(y.min() - pady, y.max() + pady)

    if show_xy_frame:
        draw_xy_frame(ax, origin=(0.12, 0.18), length=0.10, lw=2.5)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    ribbon_type = "zigzag"   # "zigzag" or "armchair"

    atoms = graphene_nanoribbon(
        n=6,
        m=10,
        type=ribbon_type,
        saturated=True,
        vacuum=6.0
    )
    atoms.center()

    plot_gnr_pretty(
        atoms,
        title="Zig Zag" if ribbon_type == "zigzag" else "Armchair",
        savepath=f"{ribbon_type}_clean.svg"
    )
