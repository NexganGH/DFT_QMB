import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from common.mpl_style import set_mpl_style

# ============================================================
# Configuration
# ============================================================
RPA_EPS_CSV = "../gpw/epsilon_vs_field.csv"   # adjust if needed
OUT_FIG = "Ueff_vs_field.pdf"

# Hartree coupling from DFT fit (reference value)
U_H_eV_per_e = 0.267023

# Choose screening type
RPA_MODE = "lfc"  # "nlfc" or "lfc"

# ============================================================
# Load RPA dielectric data
# ============================================================
def load_rpa_eps(path, mode="nlfc"):
    key = "eps_nlfc_q0" if mode.lower() == "nlfc" else "eps_lfc_q0"

    Ez, eps = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if key not in r.fieldnames:
            raise RuntimeError(f"Column '{key}' not found in {path}")
        for row in r:
            Ez.append(float(row["Ez"]))
            eps.append(float(row[key]))

    Ez = np.array(Ez)
    eps = np.array(eps)

    idx = np.argsort(Ez)
    return Ez[idx], eps[idx]

Ez, eps = load_rpa_eps(RPA_EPS_CSV, RPA_MODE)

# ============================================================
# Build U_eff(Ez)
# ============================================================
eps0 = eps[Ez == 0.0][0] if np.any(Ez == 0.0) else eps[0]

Ueff = U_H_eV_per_e * (eps0 / eps)

# ============================================================
# Plot
# ============================================================
plt.figure()
set_mpl_style()
plt.plot(Ez, Ueff, "o-", label=r"$U_{\mathrm{eff}}(E_z)$")
plt.axhline(U_H_eV_per_e, linestyle="--", color="k",
            label=r"$U_H$ (DFT reference)")

plt.xlabel(r"$E_z$ (V/\AA)")
plt.ylabel(r"$U_{\mathrm{eff}}$ (eV per $e$)")
plt.title(f"Effective Hartree from RPA")
plt.legend()
plt.tight_layout()
plt.savefig(f'../data/{OUT_FIG}', dpi=400)
plt.show()

print(f"Saved figure to {OUT_FIG}")
