import os
import csv
import re
import numpy as np

from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from external_potentials import register_with_gpaw

# ============================================================
# Run RPA for all fields to extract the dielectric function.
# ============================================================

register_with_gpaw()

# ============================================================
# Settings
# ============================================================

GPW_DIR = "../gpw_all"
OUT_CSV = "epsilon_vs_field.csv"

Nk = 18
q_list = [1.0 / Nk, 2.0 / Nk, 3.0 / Nk]

ETA = 0.01

# ============================================================
# Resume logic
# ============================================================

done = set()
if os.path.exists(OUT_CSV):
    with open(OUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(float(row["Ez"]))

# ============================================================
# CSV header
# ============================================================

if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ez",
            "eps_nlfc_q1",
            "eps_nlfc_q2",
            "eps_nlfc_q3",
            "eps_nlfc_q0",
            "eps_lfc_q0"
        ])

# ============================================================
# Main loop
# ============================================================

pattern = re.compile(r"ab_gate_plane_A([0-9]+\.[0-9]+)\.gpw")

for fname in sorted(os.listdir(GPW_DIR)):
    m = pattern.match(fname)
    if not m:
        continue

    Ez = float(m.group(1))
    if Ez in done:
        print(f"[SKIP] Ez = {Ez:.3f}")
        continue

    print("=" * 60)
    print(f"[RPA] Ez = {Ez:.3f}")

    gpw_path = os.path.join(GPW_DIR, fname)
    calc = GPAW(gpw_path, txt=None)

    df = DielectricFunction(
        calc,
        frequencies=[0.0],
        eta=ETA,
        nbands=calc.get_number_of_bands(),
        hilbert=False
    )

    eps_nlfc = []
    eps_lfc = []

    for qx in q_list:
        q = [qx, 0.0, 0.0]
        print(f"[RPA]   q = {qx:.5f}")

        df_NLFC_w, df_LFC_w = df.get_dielectric_function(q_c=q)

        eps_nlfc.append(float(np.real(df_NLFC_w[0])))
        eps_lfc.append(float(np.real(df_LFC_w[0])))

    # --------------------------------------------------------
    # q → 0 extrapolation (linear)
    # --------------------------------------------------------

    q_arr = np.array(q_list)
    eps_nlfc = np.array(eps_nlfc)
    eps_lfc = np.array(eps_lfc)

    coeff_nlfc = np.polyfit(q_arr, eps_nlfc, deg=1)
    coeff_lfc = np.polyfit(q_arr, eps_lfc, deg=1)

    eps_nlfc_q0 = coeff_nlfc[1]
    eps_lfc_q0 = coeff_lfc[1]

    # --------------------------------------------------------
    # Save immediately
    # --------------------------------------------------------

    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{Ez:.6f}",
            f"{eps_nlfc[0]:.6f}",
            f"{eps_nlfc[1]:.6f}",
            f"{eps_nlfc[2]:.6f}",
            f"{eps_nlfc_q0:.6f}",
            f"{eps_lfc_q0:.6f}"
        ])

    print(f"[SAVE] Ez = {Ez:.3f}  eps(q→0) = {eps_nlfc_q0:.3f}")

print("=" * 60)
print("[DONE] RPA sweep finished")
