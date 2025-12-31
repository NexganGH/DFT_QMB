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

# New: RPA screening CSV (produced by your RPA sweep)
RPA_EPS_CSV = os.path.join(OUT_DIR, "epsilon_vs_field.csv")  # adjust if stored elsewhere

# Output model results
OUT_MODEL_CSV = os.path.join(OUT_DIR, "self_consistent_tb_results_with_rpa.csv")

EZ_MAX = 0.02
ROUND_EZ_DECIMALS = 3

# Geometry
d_interlayer_A = 3.35  # Å

# ------------------------------------------------------------
# Hartree parameters (from your original fit)
#   Delta - Delta_ext = b0 + U_H * P_phys
# ------------------------------------------------------------
U_H_eV_per_e = 0.267023
b0_eV = 0#0.145244

# ------------------------------------------------------------
# Choose which RPA dielectric to use
#   - "nlfc": recommended for macroscopic screening mapping
#   - "lfc": robustness check (often smaller in slab supercells)
# ------------------------------------------------------------
RPA_MODE = "nlfc"  # "nlfc" or "lfc"

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


def load_rpa_eps(path):
    """
    Reads epsilon_vs_field.csv produced by the RPA pipeline.

    Expected columns (at minimum):
      Ez, eps_nlfc_q0, eps_lfc_q0
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Missing RPA screening CSV: {path}")

    Ez_list = []
    eps_list = []

    key = "eps_nlfc_q0" if RPA_MODE.lower() == "nlfc" else "eps_lfc_q0"

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if key not in r.fieldnames:
            raise RuntimeError(
                f"RPA CSV missing column '{key}'. Found: {r.fieldnames}"
            )
        for row in r:
            try:
                Ez_list.append(float(row["Ez"]))
                eps_list.append(float(row[key]))
            except Exception:
                continue

    Ez_arr = np.array(Ez_list, float)
    eps_arr = np.array(eps_list, float)

    # Sort by Ez
    idx = np.argsort(Ez_arr)
    Ez_arr = Ez_arr[idx]
    eps_arr = eps_arr[idx]

    return Ez_arr, eps_arr


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
# Self-consistency with RPA screening
#   P_phys(Delta) = P_abs(Delta) - P_abs(b0)
#   Delta = Delta_ext + b0 + U_eff(Ez) * P_phys(Delta)
# where U_eff(Ez) = U_H * eps(0)/eps(Ez)
# ============================================================
def Delta_ext(Ez):
    return float(Ez) * float(d_interlayer_A)


def build_eps_interpolator(Ez_eps, eps_vals):
    """
    Returns a function eps(Ez) with linear interpolation in Ez.
    Clamps outside range to endpoints.
    """
    Ez_eps = np.array(Ez_eps, float)
    eps_vals = np.array(eps_vals, float)

    def eps_of_Ez(Ez):
        Ez = float(Ez)
        if Ez <= Ez_eps[0]:
            return float(eps_vals[0])
        if Ez >= Ez_eps[-1]:
            return float(eps_vals[-1])
        return float(np.interp(Ez, Ez_eps, eps_vals))

    return eps_of_Ez


def solve_SC(Ez, kpatch, P_ref, Ueff):
    # initial guess near Delta ≈ Delta_ext + b0
    Delta = Delta_ext(Ez) + b0_eV

    for _ in range(SC_ITERS):
        P_abs, _ = tb_Pabs_and_gap(Delta, kpatch)
        P_phys = P_abs - P_ref
        Delta_new = Delta_ext(Ez) + b0_eV + Ueff * P_phys
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

    # Ez list from DFT gap CSV (as before)
    Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    Ez_list = Ez_list[Ez_list <= EZ_MAX]
    Ez_list = np.sort(Ez_list)

    # Build K-patch
    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)
    kpatch = K_patch(K, K_RADIUS, NK_RADIAL, NK_ANGULAR)

    # Reference shift
    P_ref, Eg_ref = tb_Pabs_and_gap(b0_eV, kpatch)
    print("[REF] Using consistent shift P_phys(Delta) = P_abs(Delta) - P_abs(b0).")
    print(f"[REF] b0 = {b0_eV:.6f} eV")
    print(f"[REF] P_ref = P_abs(b0) = {P_ref:.6e} e/cell (absolute TB)")
    print(f"[REF] Eg(b0) (patch) = {Eg_ref:.6f} eV")

    # Load RPA epsilon and build eps(Ez)
    Ez_eps, eps_vals = load_rpa_eps(RPA_EPS_CSV)
    eps_of_Ez = build_eps_interpolator(Ez_eps, eps_vals)

    # Calibrate U_eff(Ez) so that U_eff(0)=U_H (anchor at Ez=0)
    eps0 = eps_of_Ez(0.0)
    if eps0 <= 0:
        raise RuntimeError(f"Non-positive eps(0)={eps0} from {RPA_EPS_CSV}")

    def U_eff(Ez):
        e = eps_of_Ez(Ez)
        return float(U_H_eV_per_e * (eps0 / e))

    print("=" * 60)
    print(f"[RPA] Using RPA_MODE = {RPA_MODE}")
    print(f"[RPA] eps(0) = {eps0:.6f}")
    print(f"[RPA] U_eff(0) = {U_eff(0.0):.6f} (should equal U_H={U_H_eV_per_e:.6f})")

    # Run SC with field-dependent U_eff(Ez)
    results = []
    for Ez in Ez_list:
        d_ext = Delta_ext(Ez)
        Ue = U_eff(Ez)
        d_sc = solve_SC(Ez, kpatch, P_ref, Ue)

        P_abs, Eg_sc = tb_Pabs_and_gap(d_sc, kpatch)
        P_phys = P_abs - P_ref

        results.append((Ez, d_ext, d_sc, P_phys, Eg_sc, d_sc - d_ext, Ue, eps_of_Ez(Ez)))

        print(f"[SC] Ez={Ez:.4f}  eps={eps_of_Ez(Ez):.4f}  U_eff={Ue:.6f}  "
              f"Delta_ext={d_ext:.6f}  Delta_sc={d_sc:.6f}  P={P_phys:.6e}  Eg={Eg_sc:.6f}")

    # Save CSV
    with open(OUT_MODEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Ez",
            "Delta_ext_eV",
            "Delta_sc_eV",
            "P_phys_e_per_cell",
            "Eg_sc_eV",
            "Delta_minus_Delta_ext_eV",
            "U_eff_eV_per_e",
            "eps_used"
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
    Ueff_m = np.array([r[6] for r in results])
    eps_m = np.array([r[7] for r in results])

    set_mpl_style()

    # Plot: eps and Ueff vs Ez (new diagnostics)
    plt.figure()
    plt.plot(Ez_m, eps_m, "o-")
    plt.xlabel(r"$E_z$ (V/Å)")
    plt.ylabel(r"$\varepsilon(E_z)$")
    plt.title(r"RPA screening used in SC loop")
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, Ueff_m, "o-")
    plt.xlabel(r"$E_z$ (V/Å)")
    plt.ylabel(r"$U_{\mathrm{eff}}(E_z)$ (eV per e/cell)")
    plt.title(r"Effective Hartree coupling from RPA screening")
    plt.tight_layout()

    # Plot: Gap vs field (TB SC vs DFT)
    plt.figure()
    plt.plot(Ez_m, Eg_m, linestyle="-", alpha=0.4, label="_nolegend_", color="red")
    plt.plot(Ez_m, Eg_m, marker="o", linestyle="None", label=f"TB + RPA-{str.upper(RPA_MODE)}", color="red")

    mask = Ez_gap <= EZ_MAX
    if Ez_gap.size > 0:
        plt.plot(Ez_gap[mask], Eg_dft[mask], linestyle="-", alpha=0.4, label="_nolegend_", color="blue")
        plt.plot(Ez_gap[mask], Eg_dft[mask], marker="o", linestyle="None", label="DFT", color="blue")

    plt.xlabel(r"$E_z$ (V/Å)")
    plt.ylabel(r"$E_g$ (eV)")
    plt.title(f"Gap vs External Field (RPA-{str.upper(RPA_MODE)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../data/gap_vs_field_predictions_with_rpa_{RPA_MODE}.pdf", dpi=500)

    # Plot: Polarisation vs field
    plt.figure()
    plt.plot(Ez_m, P_m, "o-", label="SC TB predictions (RPA-screened)")
    if Ez_pol.size > 0:
        plt.plot(Ez_pol[Ez_pol <= EZ_MAX], P_dft[Ez_pol <= EZ_MAX], "o", label="DFT P_ind")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("P (e/cell)")
    plt.title("Polarisation vs field (shifted, P(0)=0 by construction)")
    plt.legend()
    plt.tight_layout()

    # Plot: Delta
    plt.figure()
    plt.plot(Ez_m, Delta_m, "o-", label="Delta_sc (RPA-screened)")
    plt.plot(Ez_m, Delta_ext_m + b0_eV, "--", label="Delta_ext + b0")
    plt.xlabel("Ez (V/Å)")
    plt.ylabel("Delta (eV)")
    plt.title("Internal asymmetry")
    plt.legend()
    plt.tight_layout()

    # Debug: Delta - Delta_ext vs P (now with variable Ueff, so not a straight line)
    plt.figure()
    plt.plot(P_m, Delta_minus_ext, "o", label="SC points (RPA-screened)")
    plt.xlabel("P (e/cell)")
    plt.ylabel(r"$\Delta-\Delta_{\mathrm{ext}}$ (eV)")
    plt.title("Self-consistent points with field-dependent screening")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
