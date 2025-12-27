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
OUT_MODEL_CSV = os.path.join(OUT_DIR, "self_consistent_tb_results.csv")

EZ_MAX = 0.02
ROUND_EZ_DECIMALS = 3

# Geometry
d_interlayer_A = 3.35  # Å

# ------------------------------------------------------------
# Hartree parameters (symmetric system)
# ------------------------------------------------------------
U_H_eV_per_e = 0.257
b0_eV = 0.0

# ------------------------------------------------------------
# Tight-binding parameters
# ------------------------------------------------------------
a_cc = 1.42
gamma0_eV = 2.687
gamma1_eV = 0.262

# ------------------------------------------------------------
# K-patch parameters
# ------------------------------------------------------------
K_RADIUS_GAP = 0.02
K_RADIUS_POL = 0.12

NK_RADIAL_GAP = 30
NK_ANGULAR_GAP = 90

NK_RADIAL_POL = 30
NK_ANGULAR_POL = 90

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


def K_patch(K, radius, nr, na, include_center=True):
    ks = []
    ws = []

    if include_center:
        ks.append(K.copy())
        ws.append(0.0)

    for i in range(nr):
        r = radius * (i + 0.5) / nr
        w = r
        for j in range(na):
            theta = 2.0 * np.pi * j / na
            ks.append(K + r * np.array([np.cos(theta), np.sin(theta)]))
            ws.append(w)

    return np.array(ks, float), np.array(ws, float)


# ============================================================
# TB Hamiltonian (AB bilayer)
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
# Observables
# ============================================================
def tb_gap(Delta, kpatch):
    vtop = -1e9
    cbot = +1e9
    for kx, ky in kpatch:
        evals = np.linalg.eigvalsh(H_ab(kx, ky, Delta))
        vtop = max(vtop, evals[1])
        cbot = min(cbot, evals[2])
    return max(0.0, cbot - vtop)


def tb_Pabs(Delta, kpatch, weights):
    pol = 0.0
    norm = np.sum(weights)
    for (kx, ky), w in zip(kpatch, weights):
        if w == 0:
            continue
        _, vecs = np.linalg.eigh(H_ab(kx, ky, Delta))
        for b in (0, 1):
            v = vecs[:, b]
            w1 = (np.abs(v[0])**2 + np.abs(v[1])**2)
            w2 = (np.abs(v[2])**2 + np.abs(v[3])**2)
            pol += w * (w1 - w2)
    return 2.0 * pol / norm


# ============================================================
# Self-consistency
# ============================================================
def Delta_ext(Ez):
    return Ez * d_interlayer_A


def solve_SC(Ez, kpatch_pol, w_pol, P_ref):
    Delta = Delta_ext(Ez)
    for _ in range(SC_ITERS):
        P_abs = tb_Pabs(Delta, kpatch_pol, w_pol)
        P_phys = P_abs - P_ref
        Delta_new = Delta_ext(Ez) + U_H_eV_per_e * P_phys
        if abs(Delta_new - Delta) < SC_TOL_eV:
            return Delta_new
        Delta = Delta_new
    return Delta


# ============================================================
# Main
# ============================================================
def main():
    Ez_gap, Eg_dft = load_csv_xy(GAP_CSV, "Ez", "bandgap_eV")
    Ez_pol, P_dft = load_csv_xy(POL_CSV, "Ez", "layer_polarisation_electrons")

    # --- FILTER DFT DATA ---
    mask_gap = Ez_gap <= EZ_MAX
    Ez_gap = Ez_gap[mask_gap]
    Eg_dft = Eg_dft[mask_gap]

    mask_pol = Ez_pol <= EZ_MAX
    Ez_pol = Ez_pol[mask_pol]
    P_dft = P_dft[mask_pol]

    # --- FIELD GRID ---
    Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    Ez_list = np.sort(Ez_list)

    # --- K patches ---
    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)

    kpatch_gap, _ = K_patch(K, K_RADIUS_GAP, NK_RADIAL_GAP, NK_ANGULAR_GAP, True)
    kpatch_pol, w_pol = K_patch(K, K_RADIUS_POL, NK_RADIAL_POL, NK_ANGULAR_POL, True)

    # --- reference ---
    P_ref = tb_Pabs(0.0, kpatch_pol, w_pol)

    # --- run SC ---
    results = []
    for Ez in Ez_list:
        Delta_sc = solve_SC(Ez, kpatch_pol, w_pol, P_ref)
        Eg_sc = tb_gap(Delta_sc, kpatch_gap)
        P_phys = tb_Pabs(Delta_sc, kpatch_pol, w_pol) - P_ref
        results.append((Ez, Delta_sc, P_phys, Eg_sc))
        print(f"Ez={Ez:.4f}  Delta={Delta_sc:.6f}  P={P_phys:.6e}  Eg={Eg_sc:.6f}")

    # --- save ---
    with open(OUT_MODEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ez", "Delta_sc_eV", "P_phys_e_per_cell", "Eg_sc_eV"])
        w.writerows(results)

    # --- plots ---
    Ez_m = np.array([r[0] for r in results])
    Eg_m = np.array([r[3] for r in results])
    P_m = np.array([r[2] for r in results])

    plt.figure()
    plt.plot(Ez_m, Eg_m, "o-", label="TB+Hartree SC")
    plt.plot(Ez_gap, Eg_dft, "o", label="DFT")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("Eg (eV)")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, P_m, "o-", label="TB+Hartree SC")
    plt.plot(Ez_pol, P_dft, "o", label="DFT")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("P (e/cell)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
