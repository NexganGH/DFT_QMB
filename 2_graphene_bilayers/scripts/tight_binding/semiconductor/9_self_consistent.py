import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
OUT_DIR = "../gpw"
GAP_CSV = os.path.join(OUT_DIR, "bandgaps_KZOOM.csv")
POL_CSV = os.path.join(OUT_DIR, "layer_polarisation_vs_field.csv")

EZ_MAX = 0.02
ROUND_EZ_DECIMALS = 3

# Geometry
d_interlayer_A = 3.35  # Å
DELTA_EXT_SIGN = -1.0

# ------------------------------------------------------------
# Hartree parameters (from DFT fit)
# ------------------------------------------------------------
U_H_eV_per_e = 0.257
b0_eV = 0.133168

# ------------------------------------------------------------
# Tight-binding parameters (FULL model)
# ------------------------------------------------------------
a_cc = 1.42
gamma0 = 2.687
gamma1 = 0.262
gamma3 = -0.3
gamma4 = 0.044

# ------------------------------------------------------------
# k-patches
# ------------------------------------------------------------
NK_RADIAL_P = 25
NK_ANGULAR_P = 60
K_RADIUS_P = 0.2

NK_RADIAL_G = 40
NK_ANGULAR_G = 160
K_RADIUS_G = 0.04

SC_ITERS = 200
SC_TOL = 1e-6


# ============================================================
# Lattice utilities
# ============================================================
def graphene_reciprocal_vectors(a_cc):
    a = np.sqrt(3) * a_cc
    a1 = np.array([ a/2,  np.sqrt(3)*a/2])
    a2 = np.array([-a/2,  np.sqrt(3)*a/2])
    B = 2*np.pi*np.linalg.inv(np.stack([a1, a2], axis=1)).T
    return B[:,0], B[:,1]


def K_point(b1, b2):
    return (b1 + 2*b2) / 3


def K_patch(K, R, nr, na, include_center=False):
    ks = []
    if include_center:
        ks.append(K)
    for i in range(nr):
        r = R * (i + 0.5) / nr
        for j in range(na):
            th = 2*np.pi * j / na
            ks.append(K + r*np.array([np.cos(th), np.sin(th)]))
    return np.array(ks)


# ============================================================
# TB Hamiltonian (γ0–γ1–γ3–γ4)
# ============================================================
def f_k(kx, ky):
    d = np.array([
        [0.0, a_cc],
        [ np.sqrt(3)*a_cc/2, -a_cc/2],
        [-np.sqrt(3)*a_cc/2, -a_cc/2]
    ])
    return np.sum(np.exp(1j*(d[:,0]*kx + d[:,1]*ky)))


def H_ab_full(kx, ky, Delta):
    f = f_k(kx, ky)
    fc = np.conj(f)

    D1, D2 = +0.5*Delta, -0.5*Delta

    H = np.zeros((4,4), dtype=complex)

    # onsite
    H[0,0] = H[1,1] = D1
    H[2,2] = H[3,3] = D2

    # γ0
    H[0,1] = -gamma0 * f
    H[1,0] = -gamma0 * fc
    H[2,3] = -gamma0 * f
    H[3,2] = -gamma0 * fc

    # γ1
    H[1,2] = H[2,1] = gamma1

    # γ3
    H[0,3] = -gamma3 * fc
    H[3,0] = -gamma3 * f

    # γ4
    H[0,2] = gamma4 * f
    H[2,0] = gamma4 * fc
    H[1,3] = gamma4 * f
    H[3,1] = gamma4 * fc

    return H


# ============================================================
# Observables
# ============================================================
def tb_polarisation_patch(Delta, kpatch):
    pol = 0.0
    for kx, ky in kpatch:
        _, vecs = np.linalg.eigh(H_ab_full(kx, ky, Delta))
        for b in (0,1):
            v = vecs[:,b]
            w1 = abs(v[0])**2 + abs(v[1])**2
            w2 = abs(v[2])**2 + abs(v[3])**2
            pol += (w2 - w1)
    return 2.0 * pol / len(kpatch)


def tb_gap_patch(Delta, kpatch):
    vtop, cbot = -1e9, +1e9
    for kx, ky in kpatch:
        evals = np.linalg.eigvalsh(H_ab_full(kx, ky, Delta))
        vtop = max(vtop, evals[1])
        cbot = min(cbot, evals[2])
    return max(0.0, cbot - vtop)


# ============================================================
# Self-consistency
# ============================================================
def Delta_ext(Ez):
    return DELTA_EXT_SIGN * Ez * d_interlayer_A


def solve_SC(Ez, kpatch_P, P_ref):
    Delta = Delta_ext(Ez) + b0_eV
    for _ in range(SC_ITERS):
        P = tb_polarisation_patch(Delta, kpatch_P) - P_ref
        Delta_new = Delta_ext(Ez) + b0_eV + U_H_eV_per_e * P
        if abs(Delta_new - Delta) < SC_TOL:
            break
        Delta = Delta_new
    return Delta


# ============================================================
# Main
# ============================================================
def main():
    Ez_gap, Eg_dft = load_csv_xy(GAP_CSV, "Ez", "bandgap_eV")
    Ez_pol, P_dft  = load_csv_xy(POL_CSV, "Ez", "layer_polarisation_electrons")

    Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    Ez_list = Ez_list[Ez_list <= EZ_MAX]

    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)

    kpatch_P = K_patch(K, K_RADIUS_P, NK_RADIAL_P, NK_ANGULAR_P)
    kpatch_G = K_patch(K, K_RADIUS_G, NK_RADIAL_G, NK_ANGULAR_G, include_center=True)

    # SAME SHIFTING AS BEFORE
    P_ref = tb_polarisation_patch(b0_eV, kpatch_P)

    results = []
    for Ez in Ez_list:
        Delta = solve_SC(Ez, kpatch_P, P_ref)
        P = tb_polarisation_patch(Delta, kpatch_P) - P_ref
        Eg = tb_gap_patch(Delta, kpatch_G)
        results.append((Ez, Delta, P, Eg))
        print(f"[SC] Ez={Ez:.4f}  Δ={Delta:.4f}  P={P:.4e}  Eg={Eg:.4f}")

    # plots
    Ez_m = np.array([r[0] for r in results])
    P_m  = np.array([r[2] for r in results])
    Eg_m = np.array([r[3] for r in results])

    plt.figure()
    plt.plot(Ez_m, Eg_m, "o-", label="TB+Hartree (full)")
    m = Ez_gap <= EZ_MAX
    plt.plot(Ez_gap[m], Eg_dft[m], "o", label="DFT")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("Eg (eV)")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, P_m, "o-", label="TB+Hartree (full)")
    plt.plot(Ez_pol, P_dft, "o", label="DFT")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("P (e/cell)")
    plt.legend()
    plt.tight_layout()

    plt.show()


def load_csv_xy(path, xkey, ykey):
    x, y = [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x.append(float(row[xkey]))
                y.append(float(row[ykey]))
            except:
                pass
    return np.array(x), np.array(y)


if __name__ == "__main__":
    main()
