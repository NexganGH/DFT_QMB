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
# Hartree parameters (from your fit)
#   Delta - Delta_ext = b0 + U_H * P_phys
# with P_phys defined so that P_phys(Ez=0)=0 => Delta_sc(0)=b0
# ------------------------------------------------------------
U_H_eV_per_e = 0.257
b0_eV = 0.133168

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
    return np.array(ks, float)


# ============================================================
# TB Hamiltonian (AB bilayer, gamma0-gamma1, bias +/-Delta/2)
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

    # interlayer dimer hopping (B1 <-> A2)
    H[1, 2] = gamma1_eV
    H[2, 1] = gamma1_eV
    return H


# ============================================================
# Polarisation + gap (absolute TB quantity)
# ============================================================
def tb_Pabs_and_gap(Delta, kpatch):
    pol = 0.0
    vtop = -1e9
    cbot = +1e9

    for kx, ky in kpatch:
        evals, vecs = np.linalg.eigh(H_ab(kx, ky, Delta))

        # neutrality: occupy 2 lowest bands (per spin)
        for b in (0, 1):
            v = vecs[:, b]
            w1 = (np.abs(v[0])**2 + np.abs(v[1])**2)
            w2 = (np.abs(v[2])**2 + np.abs(v[3])**2)
            pol += (w2 - w1)

        vtop = max(vtop, evals[1])
        cbot = min(cbot, evals[2])

    P_abs = 2.0 * pol / len(kpatch)  # spin degeneracy
    Eg = max(0.0, cbot - vtop)
    return float(P_abs), float(Eg)


# ============================================================
# Self-consistency with a consistent polarisation shift
#   P_phys(Delta) = P_abs(Delta) - P_abs(b0)
#   Delta = Delta_ext + b0 + U_H * P_phys(Delta)
# ============================================================
def Delta_ext(Ez):
    return float(Ez) * float(d_interlayer_A)


def solve_SC(Ez, kpatch, P_ref):
    # use the physically expected branch near Delta ≈ Delta_ext + b0
    Delta = Delta_ext(Ez) + b0_eV

    for _ in range(SC_ITERS):
        P_abs, _ = tb_Pabs_and_gap(Delta, kpatch)
        P_phys = P_abs - P_ref
        Delta_new = Delta_ext(Ez) + b0_eV + U_H_eV_per_e * P_phys
        if abs(Delta_new - Delta) < SC_TOL_eV:
            Delta = Delta_new
            break
        Delta = Delta_new

    return float(Delta)


# ============================================================
# Main
# ============================================================
def main():
    Ez_gap, Eg_dft = load_csv_xy(GAP_CSV, "Ez", "bandgap_eV")
    Ez_pol, P_dft = load_csv_xy(POL_CSV, "Ez", "layer_polarisation_electrons")

    if Ez_gap.size == 0:
        raise RuntimeError(f"Missing or empty: {GAP_CSV}")

    Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    Ez_list = Ez_list[Ez_list <= EZ_MAX]
    Ez_list = np.sort(Ez_list)

    # Build K-patch
    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)
    kpatch = K_patch(K, K_RADIUS, NK_RADIAL, NK_ANGULAR)

    # --- CONSISTENT REFERENCE SHIFT ---
    # Enforce: P_phys(Ez=0) = 0  AND Hartree fit uses the same b0.
    # That implies Delta_sc(0) = b0, so the correct reference is P_ref = P_abs(b0).
    P_ref, Eg_ref = tb_Pabs_and_gap(b0_eV, kpatch)
    print("[REF] Using consistent shift P_phys(Delta) = P_abs(Delta) - P_abs(b0).")
    print(f"[REF] b0 = {b0_eV:.6f} eV")
    print(f"[REF] P_ref = P_abs(b0) = {P_ref:.6e} e/cell (absolute TB)")
    print(f"[REF] Eg(b0) (patch) = {Eg_ref:.6f} eV")

    # Run SC
    results = []
    for Ez in Ez_list:
        d_ext = Delta_ext(Ez)
        d_sc = solve_SC(Ez, kpatch, P_ref)

        P_abs, Eg_sc = tb_Pabs_and_gap(d_sc, kpatch)
        P_phys = P_abs - P_ref

        results.append((Ez, d_ext, d_sc, P_phys, Eg_sc, d_sc - d_ext))

        print(f"[SC] Ez={Ez:.4f}  Delta_ext={d_ext:.6f}  Delta_sc={d_sc:.6f}  "
              f"P={P_phys:.6e}  Eg={Eg_sc:.6f}")

    # Save CSV
    with open(OUT_MODEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Ez", "Delta_ext_eV", "Delta_sc_eV", "P_phys_e_per_cell", "Eg_sc_eV",
            "Delta_minus_Delta_ext_eV"
        ])
        for row in results:
            w.writerow(row)

    print(f"\nSaved results to: {OUT_MODEL_CSV}")

    # Arrays
    Ez_m = np.array([r[0] for r in results])
    Delta_ext_m = np.array([r[1] for r in results])
    Delta_m = np.array([r[2] for r in results])
    P_m = np.array([r[3] for r in results])
    Eg_m = np.array([r[4] for r in results])
    Delta_minus_ext = np.array([r[5] for r in results])

    # Plots
    plt.figure()
    plt.plot(Ez_m, Eg_m, "o-", label="TB+Hartree SC Eg")
    if Ez_gap.size > 0:
        plt.plot(Ez_gap[Ez_gap <= EZ_MAX], Eg_dft[Ez_gap <= EZ_MAX], "o", label="DFT Eg (KZOOM)")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("Eg (eV)")
    plt.title("Gap vs field")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, P_m, "o-", label="TB+Hartree SC P (shifted)")
    if Ez_pol.size > 0:
        plt.plot(Ez_pol[Ez_pol <= EZ_MAX], P_dft[Ez_pol <= EZ_MAX], "o", label="DFT P_ind")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("P (e/cell)")
    plt.title("Polarisation vs field (shifted, P(0)=0 by construction)")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, Delta_m, "o-", label="Delta_sc")
    plt.plot(Ez_m, Delta_ext_m + b0_eV, "--", label="Delta_ext + b0")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("Delta (eV)")
    plt.title("Internal asymmetry")
    plt.legend()
    plt.tight_layout()

    # Debug: Delta - Delta_ext vs P (should match b0 + U_H P)
    plt.figure()
    plt.plot(P_m, Delta_minus_ext, "o", label="SC points")
    Pfit = np.linspace(P_m.min(), P_m.max(), 200)
    plt.plot(Pfit, b0_eV + U_H_eV_per_e * Pfit, "--",
             label=rf"fit: $b_0 + U_H P$ (U_H={U_H_eV_per_e:.3f})")
    plt.xlabel("P (e/cell)")
    plt.ylabel(r"$\Delta-\Delta_{\mathrm{ext}}$ (eV)")
    plt.title("Consistency with fitted Hartree law (debug)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
