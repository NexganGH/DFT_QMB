import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from common.mpl_style import set_mpl_style

# ============================================================
# Configuration
# ============================================================
OUT_DIR = "../gpw"
GAP_CSV = os.path.join(OUT_DIR, "bandgaps_KZOOM.csv")
POL_CSV = os.path.join(OUT_DIR, "layer_polarisation_vs_field.csv")
OUT_MODEL_CSV = os.path.join(OUT_DIR, "self_consistent_tb_results.csv")

EZ_MAX = 0.02
ROUND_EZ_DECIMALS = 3

# Fields for band-structure panels
EZ_BANDS = [0.0, 0.002, 0.008, 0.020]

# Geometry
d_interlayer_A = 3.35  # Å

# ------------------------------------------------------------
# Hartree parameters
# ------------------------------------------------------------
U_H_eV_per_e = 0.267023
b0_eV = 0.145244

# ------------------------------------------------------------
# Tight-binding parameters
# ------------------------------------------------------------
a_cc = 1.42
gamma0_eV = 2.687
gamma1_eV = 0.262

# K-patch sampling
NK_RADIAL = 25
NK_ANGULAR = 60
K_RADIUS = 0.08  # 1/Å

# Self-consistency
SC_TOL_eV = 1e-6
SC_ITERS = 120


# ============================================================
# CSV utilities
# ============================================================
def load_csv_xy(path, xkey, ykey):
    x, y = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x.append(float(row[xkey]))
                y.append(float(row[ykey]))
            except Exception:
                continue
    return np.array(x, float), np.array(y, float)


# ============================================================
# Lattice + reciprocal space
# ============================================================
def graphene_reciprocal_vectors(a_cc_):
    a = np.sqrt(3.0) * a_cc_
    a1 = np.array([a / 2.0,  np.sqrt(3.0) * a / 2.0])
    a2 = np.array([-a / 2.0, np.sqrt(3.0) * a / 2.0])
    A = np.stack([a1, a2], axis=1)
    B = 2.0 * np.pi * np.linalg.inv(A).T
    return B[:, 0], B[:, 1]


def K_point(b1, b2):
    return (b1 + 2.0 * b2) / 3.0


def K_patch(K, radius, nr, na):
    ks = []
    for i in range(nr):
        r = radius * (i + 0.5) / nr
        for j in range(na):
            theta = 2.0 * np.pi * j / na
            ks.append(K + r * np.array([np.cos(theta), np.sin(theta)]))
    return np.array(ks, float)


# ============================================================
# TB Hamiltonian
# ============================================================
def f_k(kx, ky):
    d1 = np.array([0.0, a_cc])
    d2 = np.array([np.sqrt(3.0) * a_cc / 2.0, -a_cc / 2.0])
    d3 = np.array([-np.sqrt(3.0) * a_cc / 2.0, -a_cc / 2.0])
    return (
        np.exp(1j * (kx * d1[0] + ky * d1[1])) +
        np.exp(1j * (kx * d2[0] + ky * d2[1])) +
        np.exp(1j * (kx * d3[0] + ky * d3[1]))
    )


def H_ab(kx, ky, Delta):
    fk = f_k(kx, ky)
    t = -gamma0_eV * fk
    D1, D2 = +0.5 * Delta, -0.5 * Delta

    H = np.zeros((4, 4), dtype=complex)

    H[0, 0] = H[1, 1] = D1
    H[2, 2] = H[3, 3] = D2

    H[0, 1] = t
    H[1, 0] = np.conjugate(t)
    H[2, 3] = t
    H[3, 2] = np.conjugate(t)

    H[1, 2] = gamma1_eV
    H[2, 1] = gamma1_eV
    return H


# ============================================================
# Polarisation + gap
# ============================================================
def tb_Pabs_and_gap(Delta, kpatch):
    pol = 0.0
    vtop = -1e9
    cbot = +1e9

    for kx, ky in kpatch:
        evals, vecs = np.linalg.eigh(H_ab(kx, ky, Delta))

        for b in (0, 1):
            v = vecs[:, b]
            w1 = (np.abs(v[0])**2 + np.abs(v[1])**2)
            w2 = (np.abs(v[2])**2 + np.abs(v[3])**2)
            pol += (w2 - w1)

        vtop = max(vtop, evals[1])
        cbot = min(cbot, evals[2])

    P_abs = 2.0 * pol / len(kpatch)
    Eg = max(0.0, cbot - vtop)
    return float(P_abs), float(Eg)


# ============================================================
# Self-consistency
# ============================================================
def Delta_ext(Ez):
    return float(Ez) * float(d_interlayer_A)


def solve_SC(Ez, kpatch, P_ref):
    Delta = Delta_ext(Ez) + b0_eV
    for _ in range(SC_ITERS):
        P_abs, _ = tb_Pabs_and_gap(Delta, kpatch)
        P_phys = P_abs - P_ref
        Delta_new = Delta_ext(Ez) + b0_eV + U_H_eV_per_e * P_phys
        if abs(Delta_new - Delta) < SC_TOL_eV:
            return float(Delta_new)
        Delta = Delta_new
    return float(Delta)


# ============================================================
# Band structure helpers (zoom around K)
# ============================================================
def k_cut_around_K(K, qmax=0.05, nk=400):
    qs = np.linspace(-qmax, qmax, nk)
    ks = np.array([K + np.array([q, 0.0]) for q in qs])
    return qs, ks


def tb_bands_along_cut(ks, Delta):
    bands = []
    for kx, ky in ks:
        bands.append(np.linalg.eigvalsh(H_ab(kx, ky, Delta)))
    return np.array(bands)


# ============================================================
# Main
# ============================================================
def main():
    Ez_gap, _ = load_csv_xy(GAP_CSV, "Ez", "bandgap_eV")

    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)
    kpatch = K_patch(K, K_RADIUS, NK_RADIAL, NK_ANGULAR)

    P_ref, _ = tb_Pabs_and_gap(b0_eV, kpatch)

    Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    Ez_list = Ez_list[Ez_list <= EZ_MAX]

    results = []
    for Ez in Ez_list:
        Delta_sc = solve_SC(Ez, kpatch, P_ref)
        results.append((Ez, Delta_sc))

    Ez_m = np.array([r[0] for r in results])
    Delta_m = np.array([r[1] for r in results])

    # ========================================================
    # Zoomed band structures around K (2x2 grid)
    # ========================================================
    set_mpl_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, Ez_sel in zip(axes, EZ_BANDS):
        idx = np.argmin(np.abs(Ez_m - Ez_sel))
        Delta_sel = Delta_m[idx]

        qs, ks = k_cut_around_K(K, qmax=0.05, nk=400)
        bands = tb_bands_along_cut(ks, Delta_sel)

        Ef = 0.5 * (bands[:, 1].max() + bands[:, 2].min())
        bands -= Ef

        for n in range(4):
            ax.plot(qs, bands[:, n], lw=2)

        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_title(rf"$E_z = {Ez_sel:.3f}\ \mathrm{{V/\AA}}$")

    axes[2].set_xlabel(r"$k - K$ ($\mathrm{\AA^{-1}}$)")
    axes[3].set_xlabel(r"$k - K$ ($\mathrm{\AA^{-1}}$)")
    axes[0].set_ylabel(r"$E - E_F$ (eV)")
    axes[2].set_ylabel(r"$E - E_F$ (eV)")

    plt.suptitle("Bilayer graphene band structure under external field", y=0.98)
    plt.tight_layout()
    plt.savefig("../data/tb_bands_zoom_K_grid.pdf", dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
