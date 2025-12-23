import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from itertools import combinations
from common.mpl_style import set_mpl_style


# =================================================
# Graphene monolayer with indices retained
# =================================================
def graphene_monolayer_indexed(nx=6, ny=4, a_cc=1.0):
    """
    Returns:
      A, B: dict[(i,j)] -> np.array([x,y])
      bonds: np.array shape (M,2,2) segments between A(i,j) and neighbour B's
      a1, a2, d: lattice vectors and A->B basis vector
    """
    a1 = np.array([np.sqrt(3) * a_cc, 0.0])
    a2 = np.array([np.sqrt(3) / 2 * a_cc, 3 / 2 * a_cc])
    d  = np.array([np.sqrt(3) / 2 * a_cc, 1 / 2 * a_cc])

    A, B = {}, {}
    for i in range(nx):
        for j in range(ny):
            R = i * a1 + j * a2
            A[(i, j)] = R
            B[(i, j)] = R + d

    # Nearest-neighbour bonds (A connects to 3 B's)
    segs = []
    for (i, j), rA in A.items():
        for keyB in [(i, j), (i - 1, j), (i, j - 1)]:
            if keyB in B:
                segs.append([rA, B[keyB]])

    bonds = np.array(segs, dtype=float)
    return A, B, bonds, a1, a2, d


# =================================================
# Hexagon detection (pure geometry)
# =================================================
def find_hexagons_from_points(points, a_cc, tol=1e-3):
    """
    points: (N,2) array of all atom positions in one layer
    Returns list of hexagons as (6,2) arrays (sorted CCW)
    """
    hexagons = []
    for i, j, k in combinations(range(len(points)), 3):
        center = (points[i] + points[j] + points[k]) / 3
        dists = np.linalg.norm(points - center, axis=1)
        mask = np.abs(dists - a_cc) < tol
        if np.count_nonzero(mask) == 6:
            h = points[mask]
            angles = np.arctan2(h[:, 1] - center[1], h[:, 0] - center[0])
            hexagons.append(h[np.argsort(angles)])

    # remove duplicates
    unique = []
    for h in hexagons:
        if not any(np.allclose(h, u) for u in unique):
            unique.append(h)
    return unique


def atoms_from_hexagons(hexagons, tol=1e-6):
    atoms = set()
    for h in hexagons:
        for p in h:
            atoms.add(tuple(np.round(p / tol).astype(int)))
    return atoms


def bonds_from_hexagons(hexagons, tol=1e-6):
    bonds = set()
    for h in hexagons:
        for i in range(6):
            p1 = tuple(np.round(h[i] / tol).astype(int))
            p2 = tuple(np.round(h[(i + 1) % 6] / tol).astype(int))
            bonds.add(tuple(sorted((p1, p2))))
    return bonds


def filter_bonds_by_hexagons(bonds, valid_bonds_set, tol=1e-6):
    out = []
    for b in bonds:
        p1 = tuple(np.round(b[0] / tol).astype(int))
        p2 = tuple(np.round(b[1] / tol).astype(int))
        if tuple(sorted((p1, p2))) in valid_bonds_set:
            out.append(b)
    return np.array(out)


def filter_dict_points_by_valid_atoms(D, valid_atoms_set, tol=1e-6):
    """
    Keep only dict entries whose coordinate appears in valid_atoms_set.
    """
    out = {}
    for key, p in D.items():
        kp = tuple(np.round(p / tol).astype(int))
        if kp in valid_atoms_set:
            out[key] = p
    return out


# =================================================
# Plot AB bilayer with primitive-cell-correct labels
# =================================================
def plot_ab_bilayer(nx=7, ny=5, a_cc=1.0):

    # Bottom layer
    A1, B1, bonds1, a1, a2, d = graphene_monolayer_indexed(nx, ny, a_cc)

    # AB shift: choose shift = d so that A2(i,j) coincides with B1(i,j)
    shift = d.copy()

    # Top layer (same indices, shifted)
    A2 = {k: v + shift for k, v in A1.items()}
    B2 = {k: v + shift for k, v in B1.items()}
    bonds2 = bonds1 + shift

    # Build point clouds for hexagon detection
    pts1 = np.vstack([np.vstack(list(A1.values())), np.vstack(list(B1.values()))])
    pts2 = np.vstack([np.vstack(list(A2.values())), np.vstack(list(B2.values()))])

    hex1 = find_hexagons_from_points(pts1, a_cc)
    hex2 = find_hexagons_from_points(pts2, a_cc)

    # Keep only atoms/bonds belonging to full hexagons (clean boundaries)
    valid_atoms_1 = atoms_from_hexagons(hex1)
    valid_atoms_2 = atoms_from_hexagons(hex2)
    valid_bonds_1 = bonds_from_hexagons(hex1)
    valid_bonds_2 = bonds_from_hexagons(hex2)

    A1 = filter_dict_points_by_valid_atoms(A1, valid_atoms_1)
    B1 = filter_dict_points_by_valid_atoms(B1, valid_atoms_1)
    A2 = filter_dict_points_by_valid_atoms(A2, valid_atoms_2)
    B2 = filter_dict_points_by_valid_atoms(B2, valid_atoms_2)

    bonds1 = filter_bonds_by_hexagons(bonds1, valid_bonds_1)
    bonds2 = filter_bonds_by_hexagons(bonds2, valid_bonds_2)

    # -------------------------------------------------
    # Choose a central primitive cell (i,j) that survived trimming
    # We pick the (i,j) whose dimer B1(i,j)=A2(i,j) is closest to the geometric centre.
    # -------------------------------------------------
    if len(B1) == 0 or len(A2) == 0 or len(B2) == 0 or len(A1) == 0:
        raise RuntimeError("Trimming removed all atoms. Increase nx, ny or relax trimming.")

    all_bottom = np.vstack([np.vstack(list(A1.values())), np.vstack(list(B1.values()))])
    centre = all_bottom.mean(axis=0)

    common_keys = set(A1.keys()) & set(B1.keys()) & set(A2.keys()) & set(B2.keys())
    if not common_keys:
        raise RuntimeError("No complete (i,j) cell survived trimming. Increase nx, ny.")

    key0 = min(common_keys, key=lambda k: np.linalg.norm(B1[k] - centre))

    # Primitive-cell atoms (correct by construction)
    A1_site = A1[key0]
    B1_site = B1[key0]
    A2_site = A2[key0]   # coincides with B1_site in-plane
    B2_site = B2[key0]   # THIS is the “top-right B2” of the same primitive cell

    dimer_site = 0.5 * (A2_site + B1_site)  # same point

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    for h in hex1:
        ax.add_patch(Polygon(h, facecolor="#2A9D8F", alpha=0.25, edgecolor="none", zorder=0))
    for h in hex2:
        ax.add_patch(Polygon(h, facecolor="#E9C46A", alpha=0.25, edgecolor="none", zorder=1))

    ax.add_collection(LineCollection(bonds1, colors="#2A9D8F", linewidths=2.5, alpha=0.7, zorder=2))
    ax.add_collection(LineCollection(bonds2, colors="#E9C46A", linewidths=2.5, alpha=0.7, zorder=3))

    s = 180
    A1_arr = np.vstack(list(A1.values()))
    B1_arr = np.vstack(list(B1.values()))
    A2_arr = np.vstack(list(A2.values()))
    B2_arr = np.vstack(list(B2.values()))

    ax.scatter(A1_arr[:, 0], A1_arr[:, 1], s=s, c="#2A9D8F", edgecolors="k", zorder=4)
    ax.scatter(B1_arr[:, 0], B1_arr[:, 1], s=s, c="#2A9D8F", edgecolors="k", zorder=4)
    ax.scatter(A2_arr[:, 0], A2_arr[:, 1], s=s, c="#E9C46A", edgecolors="k", zorder=5)
    ax.scatter(B2_arr[:, 0], B2_arr[:, 1], s=s, c="#E9C46A", edgecolors="k", zorder=5)

    # -------------------------------------------------
    # Labels (Bernal AB): dimer A2 ≡ B1, plus A1 and B2 in the SAME primitive cell
    # -------------------------------------------------
    # ax.text(*(dimer_site + np.array([0.0, 0.55 * a_cc])),
    #         r"$A_2 \equiv B_1$",
    #         fontsize=18, weight="bold", ha="center", va="center", zorder=10)
    #
    # ax.text(*(A1_site + np.array([-0.55 * a_cc, 0.0])),
    #         r"$A_1$",
    #         fontsize=16, weight="bold", ha="center", va="center", zorder=10)
    #
    # ax.text(*(B2_site + np.array([0.55 * a_cc, 0.0])),
    #         r"$B_2$",
    #         fontsize=16, weight="bold", ha="center", va="center", zorder=10)

    # -------------------------------------------------
    # Legend (bigger)
    # -------------------------------------------------
    ax.scatter([], [], s=360, c="#2A9D8F", edgecolors="k", label="Bottom layer")
    ax.scatter([], [], s=360, c="#E9C46A", edgecolors="k", label="Top layer")
    ax.legend(loc="upper left", fontsize=18, frameon=False)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig('out/ab_bilayer.png', dpi=500)
    plt.show()


if __name__ == "__main__":
    plot_ab_bilayer(nx=5, ny=4, a_cc=1.0)
