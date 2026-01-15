import numpy as np
from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from external_potentials import register_with_gpaw
# ===========================================================
# This is just for testing purposes to check if RPA works for one field.
# ===========================================================

register_with_gpaw()

# ------------------------------------------------------------
# Load RESPONSE-READY ground state
# ------------------------------------------------------------
gpw_file = "../gpw_all/ab_gate_plane_A0.000.gpw"
calc = GPAW(gpw_file, txt=None)

print("[RPA] Loaded:", gpw_file)
print("[RPA] Number of bands:", calc.get_number_of_bands())

# ------------------------------------------------------------
# Static RPA dielectric function
# ------------------------------------------------------------
df = DielectricFunction(
    calc,
    frequencies=[0.0],          # ω = 0 → static screening
    eta=0.01,                   # small broadening
    nbands=calc.get_number_of_bands(),
    hilbert=False
)

# ------------------------------------------------------------
# q-point commensurate with k-grid (18×18×1)
# ------------------------------------------------------------
Nk = 18
q = [1.0 / Nk, 0.0, 0.0]

print(f"[RPA] Computing ε(q) at q = {q}")

df_NLFC_w, df_LFC_w = df.get_dielectric_function(q_c=q)

# Static (ω = 0), macroscopic (G = G' = 0)
eps = df_NLFC_w[0]


print(f"[RPA] ε(q) = {eps:.6f}")
print("NLFC:", df_NLFC_w, "shape", df_NLFC_w.shape)
print(" LFC:", df_LFC_w,  "shape", df_LFC_w.shape)