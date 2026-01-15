import os
import csv
import numpy as np
import matplotlib.pyplot as plt

#=============================================================
# In this file we extract the polarization to later use for fitting.
#============================================================

# ============================================================
# Configuration
# ============================================================
OUT_DIR = "../gpw"

GAP_CSV = os.path.join(OUT_DIR, "bandgaps_KZOOM.csv")
POL_CSV = os.path.join(OUT_DIR, "layer_polarisation_vs_field.csv")

# Valid regime (as established earlier)
EZ_MAX = 0.02


# ============================================================
# I/O utilities
# ============================================================
def load_gap_csv(path):
    Ez, Eg = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ez.append(float(row["Ez"]))
            Eg.append(float(row["bandgap_eV"]))
    return np.array(Ez), np.array(Eg)


def load_polarisation_csv(path):
    Ez, P = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ez.append(float(row["Ez"]))
            P.append(float(row["layer_polarisation_electrons"]))
    return np.array(Ez), np.array(P)


def merge_by_field(Ez1, A1, Ez2, A2, tol=1e-6):
    """
    Merge two datasets by Ez value.
    """
    out_Ez, out_A1, out_A2 = [], [], []

    for e1, a1 in zip(Ez1, A1):
        mask = np.abs(Ez2 - e1) < tol
        if np.any(mask):
            out_Ez.append(e1)
            out_A1.append(a1)
            out_A2.append(A2[mask][0])

    return np.array(out_Ez), np.array(out_A1), np.array(out_A2)


# ============================================================
# Analysis
# ============================================================
def main():
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    Ez_g, Eg = load_gap_csv(GAP_CSV)
    Ez_p, P = load_polarisation_csv(POL_CSV)

    Ez, Eg, P = merge_by_field(Ez_g, Eg, Ez_p, P)

    # Restrict to valid regime
    mask = Ez <= EZ_MAX
    Ez = Ez[mask]
    Eg = Eg[mask]
    P = P[mask]

    print(f"Using {len(Ez)} points in valid regime (Ez ≤ {EZ_MAX})")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    plt.figure()
    plt.plot(Ez, Eg, "o-")
    plt.xlabel("E_z")
    plt.ylabel("Band gap E_g (eV)")
    plt.title("DFT band gap vs electric field")
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez, P, "o-")
    plt.xlabel("E_z")
    plt.ylabel("Layer polarisation P (e / cell)")
    plt.title("Induced layer polarisation vs field")
    plt.tight_layout()

    plt.figure()
    plt.plot(P, Eg, "o")
    plt.xlabel("Layer polarisation P (e / cell)")
    plt.ylabel("Band gap E_g (eV)")
    plt.title("Gap vs polarisation (small-field regime)")
    plt.tight_layout()

    # --------------------------------------------------------
    # Extract electrostatic coupling U_H
    # --------------------------------------------------------
    # Linear model:  Eg ≈ U_H * P   (in the small-field regime)
    coeffs = np.polyfit(P, Eg, 1)
    U_H = coeffs[0]

    print(f"\nExtracted electrostatic coupling:")
    print(f"  U_H ≈ {U_H:.4f} eV per electron")

    # Plot fit
    P_fit = np.linspace(P.min(), P.max(), 200)
    Eg_fit = U_H * P_fit

    plt.figure()
    plt.plot(P, Eg, "o", label="DFT data")
    plt.plot(P_fit, Eg_fit, "--", label=f"Linear fit (U_H={U_H:.3f} eV)")
    plt.xlabel("Layer polarisation P (e / cell)")
    plt.ylabel("Band gap E_g (eV)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
