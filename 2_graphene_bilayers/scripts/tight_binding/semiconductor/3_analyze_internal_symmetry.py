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
MERGED_CSV = os.path.join(OUT_DIR, "gap_polarisation_delta_merged.csv")

EZ_MAX = 0.02
ROUND_EZ_DECIMALS = 3

# ---- MANUAL FIT SELECTION ----------------------------------
# Option A: fit only points in this polarisation window
P_MIN = -0.26      # e / cell
P_MAX = -0.0075      # e / cell

# Option B: alternatively, fit only low-field points
# EZ_FIT_MAX = 0.012

# ------------------------------------------------------------
d_interlayer_A = 3.35     # Å
gamma1_eV = 0.262          # eV


# ============================================================
# CSV loaders
# ============================================================
def load_bandgaps(path):
    Ez, Eg = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                Ez.append(float(row["Ez"]))
                Eg.append(float(row["bandgap_eV"]))
            except Exception:
                continue
    return np.array(Ez), np.array(Eg)


def load_polarisation(path):
    Ez, P = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                Ez.append(float(row["Ez"]))
                P.append(float(row["layer_polarisation_electrons"]))
            except Exception:
                continue
    return np.array(Ez), np.array(P)


def merge_by_rounded_Ez(Ez1, A1, Ez2, A2, decimals=3):
    k1 = np.round(Ez1, decimals)
    k2 = np.round(Ez2, decimals)

    m1 = {float(e): a for e, a in zip(k1, A1)}
    m2 = {float(e): a for e, a in zip(k2, A2)}

    keys = sorted(set(m1) & set(m2))
    Ez = np.array(keys)
    A1m = np.array([m1[k] for k in keys])
    A2m = np.array([m2[k] for k in keys])
    return Ez, A1m, A2m


# ============================================================
# Physics
# ============================================================
def delta_from_gap(Eg, gamma1=gamma1_eV):
    Eg = np.asarray(Eg, float)
    if np.any(Eg >= 2 * gamma1):
        raise ValueError("Eg >= 2*gamma1: inversion invalid")
    return (gamma1 * Eg) / np.sqrt(4 * gamma1**2 - Eg**2)


def delta_ext_from_field(Ez, d_A=d_interlayer_A):
    return Ez * d_A


# ============================================================
# Linear regression
# ============================================================
def linear_fit_with_intercept(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a * x + b
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return a, b, rmse


# ============================================================
# Main
# ============================================================
def main():
    Ez_g, Eg = load_bandgaps(GAP_CSV)
    Ez_p, P = load_polarisation(POL_CSV)

    Ez, Eg, P = merge_by_rounded_Ez(
        Ez_g, Eg, Ez_p, P, ROUND_EZ_DECIMALS
    )

    mask = Ez <= EZ_MAX
    Ez, Eg, P = Ez[mask], Eg[mask], P[mask]

    Delta = delta_from_gap(Eg)
    Delta_ext = delta_ext_from_field(Ez)
    y = Delta - Delta_ext

    # --------------------------------------------------------
    # MANUAL FIT SELECTION
    # --------------------------------------------------------
    fit_mask = (P >= P_MIN) & (P <= P_MAX)
    # Alternative:
    # fit_mask = Ez <= EZ_FIT_MAX

    if fit_mask.sum() < 3:
        raise RuntimeError("Not enough points in selected fit range.")

    a, b, rmse = linear_fit_with_intercept(P[fit_mask], y[fit_mask])

    print("\nManual Hartree fit:")
    print(f"  Fit window: P in [{P_MIN:.3f}, {P_MAX:.3f}] e/cell")
    print(f"  U_H (slope) = {a:.6f} eV / e")
    print(f"  intercept  = {b:.6f} eV")
    print(f"  RMSE       = {rmse:.3e} eV")
    print(f"  npts       = {fit_mask.sum()}")

    # --------------------------------------------------------
    # Save merged data
    # --------------------------------------------------------
    with open(MERGED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Ez", "bandgap_eV", "P_e_per_cell",
            "Delta_eV", "Delta_ext_eV", "Delta_minus_ext_eV"
        ])
        for e, eg, p, d, de in zip(Ez, Eg, P, Delta, Delta_ext):
            w.writerow([e, eg, p, d, de, d - de])

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    set_mpl_style()
    plt.figure()
    plt.plot(Ez, Eg, "o-")
    plt.xlabel("E_z")
    plt.ylabel("E_g (eV)")
    plt.title("DFT band gap vs field")
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez, P, "o-")
    plt.xlabel("E_z")
    plt.ylabel("P (e / cell)")
    plt.title("Induced polarisation vs field")
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez, Delta, "o-", label="Delta from gap")
    plt.plot(Ez, Delta_ext, "--", label="Delta_ext")
    plt.xlabel("E_z")
    plt.ylabel("Energy (eV)")
    plt.legend()
    plt.title("Internal vs bare asymmetry")
    plt.tight_layout()

    plt.figure()
    plt.plot(P, y, "o", alpha=0.3, label="All points")
    plt.plot(P[fit_mask], y[fit_mask], "o", label="Fitted points")
    Pfit = np.linspace(P.min(), P.max(), 300)
    plt.plot(Pfit, a * Pfit + b, "--",
             label=rf"Fit: $U_H={a:.3f}$ eV/e")
    plt.xlabel("P (e / cell)")
    plt.ylabel(r"$\Delta - \Delta_{ext}$ (eV)")
    plt.title("Hartree Parameter Estimation")
    plt.legend()
    plt.tight_layout()
    plt.savefig('../data/fitted_hartree.pdf', dpi=500)

    plt.show()


if __name__ == "__main__":
    main()
