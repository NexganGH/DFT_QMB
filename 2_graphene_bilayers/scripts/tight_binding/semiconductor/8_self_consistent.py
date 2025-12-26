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
# Hartree parameters (from your DFT-derived fit)
#   Delta - Delta_ext = b0 + U_H * P
# IMPORTANT: In the model below, P is the TB-computed induced polarisation
# (shifted so that P(Ez=0)=0 by construction). No alpha scaling is used.
# ------------------------------------------------------------
U_H_eV_per_e = 0.257
b0_eV = 0.133168

# ------------------------------------------------------------
# Tight-binding parameters
# ------------------------------------------------------------
a_cc = 1.42
gamma0_eV = 2.687
gamma1_eV = 0.262

# ------------------------------------------------------------
# Two k-samplings:
#   - large patch for polarisation (needs broader k support)
#   - small, K-accurate patch for gap (needs correct band-edge region)
# ------------------------------------------------------------
# Polarisation patch (larger)
NK_RADIAL_P = 35
NK_ANGULAR_P = 90
K_RADIUS_P = 0.15  # 1/Å  (your current tuned value)

# Gap patch (smaller, more local)
NK_RADIAL_G = 45
NK_ANGULAR_G = 180
K_RADIUS_G = 0.06  # 1/Å  (tune 0.02–0.06)

# Self-consistency
SC_TOL_eV = 1e-6
SC_ITERS = 300

# Sign convention to match DFT Ez -> layer bias in TB
DELTA_EXT_SIGN = -1.0


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
    """
    Polar sampling around K. Optionally include exact center K point
    (important to avoid spurious Eg(Delta=0) > 0 due to missing K).
    """
    ks = []
    if include_center:
        ks.append(np.array(K, float))
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
# TB proxies
# ============================================================
def tb_layer_imbalance_patch(Delta, kpatch):
    """
    Patch-average layer imbalance proxy:
      P_patch = 2 * <sum_{occ} (w2 - w1)>_patch
    Spin degeneracy included.
    """
    pol = 0.0
    for kx, ky in kpatch:
        _, vecs = np.linalg.eigh(H_ab(kx, ky, Delta))
        for b in (0, 1):  # 2 occupied bands at neutrality
            v = vecs[:, b]
            w1 = (np.abs(v[0])**2 + np.abs(v[1])**2)
            w2 = (np.abs(v[2])**2 + np.abs(v[3])**2)
            pol += (w2 - w1)
    return float(2.0 * pol / len(kpatch))


def tb_gap_from_patch(Delta, kpatch_gap):
    """
    Robust-ish patch estimate of the gap: min conduction - max valence
    computed on a tight K-local patch. Includes exact K point.
    """
    vtop = -1e9
    cbot = +1e9
    for kx, ky in kpatch_gap:
        evals, _ = np.linalg.eigh(H_ab(kx, ky, Delta))
        vtop = max(vtop, float(evals[1]))
        cbot = min(cbot, float(evals[2]))
    return float(max(0.0, cbot - vtop))


# ============================================================
# Self-consistency
#   Delta = Delta_ext + b0 + U_H * P_phys(Delta)
# with P_phys(Delta) = P_patch(Delta) - P_patch(Delta0_ref)
#
# IMPORTANT choice:
# We enforce P_phys(Ez=0)=0 by defining the reference at the
# *self-consistent* Delta for Ez=0. This avoids sign/offset pathologies.
# ============================================================
def Delta_ext(Ez):
    return float(DELTA_EXT_SIGN) * float(Ez) * float(d_interlayer_A)


def solve_SC_given_Pref(Ez, kpatch_P, P_ref):
    """
    Solve SC with fixed P_ref:
      Delta = Delta_ext + b0 + U_H*(P_patch(Delta) - P_ref)
    Simple fixed-point iteration (stable for your parameter regime).
    """
    Delta = Delta_ext(Ez) + b0_eV
    for _ in range(SC_ITERS):
        P_patch = tb_layer_imbalance_patch(Delta, kpatch_P)
        P_phys = P_patch - P_ref
        Delta_new = Delta_ext(Ez) + b0_eV + U_H_eV_per_e * P_phys
        if abs(Delta_new - Delta) < SC_TOL_eV:
            return float(Delta_new)
        Delta = Delta_new
    return float(Delta)


def find_reference_at_Ez0(kpatch_P):
    """
    Determine the Ez=0 self-consistent solution, then set
    P_ref := P_patch(Delta_sc(Ez=0)) so that P_phys(Ez=0)=0 by construction.
    """
    Delta0 = solve_SC_given_Pref(0.0, kpatch_P, P_ref=0.0)
    P_ref = tb_layer_imbalance_patch(Delta0, kpatch_P)
    return float(Delta0), float(P_ref)


# ============================================================
# Main
# ============================================================
def main():
    Ez_gap, Eg_dft = load_csv_xy(GAP_CSV, "Ez", "bandgap_eV")
    Ez_pol, P_dft = load_csv_xy(POL_CSV, "Ez", "layer_polarisation_electrons")

    if Ez_gap.size == 0:
        raise RuntimeError(f"Missing or empty: {GAP_CSV}")
    if Ez_pol.size == 0:
        raise RuntimeError(f"Missing or empty: {POL_CSV}")

    # Use fields from gap CSV
    Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    Ez_list = Ez_list[Ez_list <= EZ_MAX]
    Ez_list = np.sort(Ez_list)

    # Build patches
    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)

    kpatch_P = K_patch(K, K_RADIUS_P, NK_RADIAL_P, NK_ANGULAR_P, include_center=False)
    kpatch_G = K_patch(K, K_RADIUS_G, NK_RADIAL_G, NK_ANGULAR_G, include_center=True)

    # Reference: enforce P_phys(Ez=0)=0
    Delta0, P_ref = find_reference_at_Ez0(kpatch_P)
    Eg0 = tb_gap_from_patch(Delta0, kpatch_G)

    print("[REF] Enforcing P_phys(Ez=0)=0 via P_ref = P_patch(Delta_sc(Ez=0)).")
    print(f"[REF] Delta_sc(Ez=0) = {Delta0:.6f} eV")
    print(f"[REF] P_ref          = {P_ref:.6e} (TB patch proxy)")
    print(f"[REF] Eg_sc(Ez=0)    = {Eg0:.6f} eV (gap patch)")

    # Run SC for all fields
    results = []
    for Ez in Ez_list:
        d_ext = Delta_ext(Ez)
        d_sc = solve_SC_given_Pref(Ez, kpatch_P, P_ref=P_ref)

        P_patch = tb_layer_imbalance_patch(d_sc, kpatch_P)
        P_phys = P_patch - P_ref

        Eg_sc = tb_gap_from_patch(d_sc, kpatch_G)
        results.append((Ez, d_ext, d_sc, P_phys, Eg_sc, d_sc - d_ext))

        print(f"[SC] Ez={Ez:.4f}  Delta_ext={d_ext:.6f}  Delta_sc={d_sc:.6f}  "
              f"P={P_phys:.6e}  Eg={Eg_sc:.6f}")

    # Save results
    with open(OUT_MODEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Ez", "Delta_ext_eV", "Delta_sc_eV",
            "P_tb_shifted_e_per_cell", "Eg_sc_eV",
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

    # Plots: gap
    plt.figure()
    plt.plot(Ez_m, Eg_m, "o-", label="TB+Hartree SC Eg (gap patch)")
    m = Ez_gap <= EZ_MAX
    plt.plot(Ez_gap[m], Eg_dft[m], "o", label="DFT Eg (KZOOM)")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("Eg (eV)")
    plt.title("Gap vs field")
    plt.legend()
    plt.tight_layout()

    # Plots: polarisation
    plt.figure()
    plt.plot(Ez_m, P_m, "o-", label="TB+Hartree SC P (shifted)")
    m = Ez_pol <= EZ_MAX
    plt.plot(Ez_pol[m], P_dft[m], "o", label="DFT P_ind")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("P (e/cell)")
    plt.title("Polarisation vs field (shifted, no scaling)")
    plt.legend()
    plt.tight_layout()

    # Debug: Hartree law consistency (using TB P)
    # NOTE: This is a diagnostic only. Your original fit used DFT P,
    # so do not expect perfect agreement in absolute scale.
    plt.figure()
    plt.plot(P_m, Delta_minus_ext, "o", label="SC points")
    Pfit = np.linspace(P_m.min(), P_m.max(), 200)
    plt.plot(Pfit, b0_eV + U_H_eV_per_e * Pfit, "--",
             label=rf"$b_0 + U_H P$ (U_H={U_H_eV_per_e:.3f})")
    plt.xlabel("P (e/cell)")
    plt.ylabel(r"$\Delta-\Delta_{\mathrm{ext}}$ (eV)")
    plt.title("Hartree consistency (diagnostic, TB P)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
