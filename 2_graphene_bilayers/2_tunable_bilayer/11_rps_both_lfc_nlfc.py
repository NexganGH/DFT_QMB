import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from common.mpl_style import set_mpl_style
from matplotlib.ticker import MultipleLocator

# ============================================================
# Configuration
# ============================================================
OUT_DIR = "../gpw"
GAP_CSV = os.path.join(OUT_DIR, "bandgaps_KZOOM.csv")
POL_CSV = os.path.join(OUT_DIR, "layer_polarisation_vs_field.csv")
RPA_EPS_CSV = os.path.join(OUT_DIR, "epsilon_vs_field.csv")

EZ_MAX = 0.02
ROUND_EZ_DECIMALS = 3

# Geometry
d_interlayer_A = 3.35  # Å

# Hartree parameters (DFT reference)
U_H_eV_per_e = 0.267023
b0_eV = 0.0

# Tight-binding parameters
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
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x.append(float(row[xkey]))
                y.append(float(row[ykey]))
            except Exception:
                continue
    return np.array(x), np.array(y)


def load_rpa_eps_both(path):
    Ez, eps_nlfc, eps_lfc = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                Ez.append(float(row["Ez"]))
                eps_nlfc.append(float(row["eps_nlfc_q0"]))
                eps_lfc.append(float(row["eps_lfc_q0"]))
            except Exception:
                continue
    Ez = np.array(Ez)
    idx = np.argsort(Ez)
    return Ez[idx], np.array(eps_nlfc)[idx], np.array(eps_lfc)[idx]


# ============================================================
# Lattice + K patch
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
    return np.array(ks)


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
    H[1, 2] = H[2, 1] = gamma1_eV
    return H


# ============================================================
# Polarisation + gap
# ============================================================
def tb_Pabs_and_gap(Delta, kpatch):
    pol = 0.0
    vtop, cbot = -1e9, 1e9
    for kx, ky in kpatch:
        evals, vecs = np.linalg.eigh(H_ab(kx, ky, Delta))
        for b in (0, 1):
            v = vecs[:, b]
            pol += (np.abs(v[2])**2 + np.abs(v[3])**2) \
                 - (np.abs(v[0])**2 + np.abs(v[1])**2)
        vtop = max(vtop, evals[1])
        cbot = min(cbot, evals[2])
    P_abs = 2.0 * pol / len(kpatch)
    Eg = max(0.0, cbot - vtop)
    return P_abs, Eg


# ============================================================
# Self-consistency
# ============================================================
def Delta_ext(Ez):
    return Ez * d_interlayer_A


def solve_SC(Ez, kpatch, P_ref, Ueff):
    Delta = Delta_ext(Ez)
    for _ in range(SC_ITERS):
        P_abs, _ = tb_Pabs_and_gap(Delta, kpatch)
        P_phys = P_abs - P_ref
        Delta_new = Delta_ext(Ez) + Ueff * P_phys
        if abs(Delta_new - Delta) < SC_TOL_eV:
            break
        Delta = Delta_new
    return Delta


# ============================================================
# Main
# ============================================================
def main():
    Ez_gap, Eg_dft = load_csv_xy(GAP_CSV, "Ez", "bandgap_eV")

    Ez_eps, eps_nlfc, eps_lfc = load_rpa_eps_both(RPA_EPS_CSV)

    eps0_nlfc = eps_nlfc[Ez_eps == 0.0][0]
    eps0_lfc  = eps_lfc[Ez_eps == 0.0][0]

    Ueff_nlfc = U_H_eV_per_e * eps0_nlfc / eps_nlfc
    Ueff_lfc  = U_H_eV_per_e * eps0_lfc  / eps_lfc

    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)
    kpatch = K_patch(K, K_RADIUS, NK_RADIAL, NK_ANGULAR)

    P_ref, _ = tb_Pabs_and_gap(0.0, kpatch)

    Eg_dft = Eg_dft[Ez_gap <= EZ_MAX]
    Ez_gap = Ez_gap[Ez_gap <= EZ_MAX]
    Eg_nlfc, Eg_lfc = [], []

    for Ez in Ez_gap:
        U_n = np.interp(Ez, Ez_eps, Ueff_nlfc)
        U_l = np.interp(Ez, Ez_eps, Ueff_lfc)

        D_n = solve_SC(Ez, kpatch, P_ref, U_n)
        D_l = solve_SC(Ez, kpatch, P_ref, U_l)

        Eg_nlfc.append(tb_Pabs_and_gap(D_n, kpatch)[1])
        Eg_lfc.append(tb_Pabs_and_gap(D_l, kpatch)[1])

    set_mpl_style()

    # ============================================================
    # Plot 1: Effective Hartree coupling
    # ============================================================
    plt.figure()
    plt.plot(Ez_eps, Ueff_nlfc, "o-", label="NLFC")
    plt.plot(Ez_eps, Ueff_lfc, "s--", label="LFC")
    plt.axhline(U_H_eV_per_e, linestyle=":", color="k", label=r"$U_H$")
    plt.xlabel(r"$E_z$ (V/Å)")
    plt.ylabel(r"$U_{\mathrm{eff}}$ (eV)")
    plt.legend()
    plt.tight_layout()

    plt.savefig('../gpw/rpa_effective_both.pdf', dpi=500)

    # ============================================================
    # Plot 2: Band gaps
    # ============================================================
    plt.figure()
    plt.plot(Ez_gap, Eg_nlfc, "o-", label="TB + RPA-NLFC", color='red')
    plt.plot(Ez_gap, Eg_lfc, "s--", label="TB + RPA-LFC", color='green')
    plt.plot(Ez_gap, Eg_dft, "k.", label="DFT", color="blue")
    plt.xlabel(r"$E_z$ (V/Å)")
    plt.ylabel(r"$E_g$ (eV)")
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  # major ticks every 0.05 eV
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # minor ticks every 0.01 eV

    plt.savefig('../gpw/rpa_gaps_both.pdf', dpi=500)

    plt.show()


if __name__ == "__main__":
    main()
