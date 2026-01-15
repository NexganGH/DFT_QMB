import os
import csv
import numpy as np

from gpaw import restart
from external_potentials import ChargedPlanePotential, register_with_gpaw
from gpaw_hooks import apply_external

# ===========================================================
# To obtian the dielectric functions we needed to recompute the GPW files including the plane waves, which were not
# included when doing the normal field sweep. These new .gpw files are saved in `./gpw_all`
# ===========================================================
#

register_with_gpaw()

# ============================================================
# Restart base (no field) calculation
# ============================================================

gpw_path = "../gpw/ab.gpw"  # base, zero-field calculation
atoms, calc0 = restart(gpw_path)

# ============================================================
# User settings
# ============================================================

out_dir = "../gpw_all"
os.makedirs(out_dir, exist_ok=True)

sweep_csv = os.path.join(out_dir, "field_sweep_response_ready.csv")

# Full sweep (0.000 → 0.020)
fields = np.round(np.linspace(0.000, 0.020, 21), 3)
#fields = np.array([0.000, 0.001, 0.002, 0.010, 0.020])

# --- FOR TESTING ---
# fields = [0.000, 0.010, 0.020]

z_plane = 0.0   # same as in your original runs

# ============================================================
# Load existing results (resume-safe)
# ============================================================

existing = {}
if os.path.exists(sweep_csv):
    print("[INFO] Reading existing CSV for resume")
    with open(sweep_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[np.round(float(row["Ez"]), 3)] = row["gpw_path"]
else:
    print("[INFO] No existing CSV found, starting fresh")

# ============================================================
# Prepare CSV header if needed
# ============================================================

if not os.path.exists(sweep_csv):
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ez", "gpw_path"])

# ============================================================
# Main sweep loop
# ============================================================

for Ez in fields:
    Ez = float(np.round(Ez, 3))

    if Ez in existing:
        print(f"[SKIP] Ez = {Ez:.3f} already completed")
        continue

    print("=" * 60)
    print(f"[RUN] Ez = {Ez:.3f} eV/Å")

    # --------------------------------------------------------
    # External gate potential (exactly as before)
    # --------------------------------------------------------
    ext = ChargedPlanePotential(A=Ez, z_plane=z_plane)

    # --------------------------------------------------------
    # Apply external field on top of base calculator
    # --------------------------------------------------------
    calcE = apply_external(
        calc0,
        atoms,
        ext,
        txt=os.path.join(out_dir, f"gate_A{Ez:.3f}.txt")
    )

    # Trigger SCF
    _ = atoms.get_potential_energy()

    # --------------------------------------------------------
    # Write RESPONSE-READY restart (thisPAW/GW compatible)
    # --------------------------------------------------------
    gpw_Ez = os.path.join(out_dir, f"ab_gate_plane_A{Ez:.3f}.gpw")
    calcE.write(gpw_Ez, mode="all")

    # --------------------------------------------------------
    # Append to CSV immediately (interruption-safe)
    # --------------------------------------------------------
    with open(sweep_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{Ez:.6f}", gpw_Ez])

    print(f"[SAVE] Ez = {Ez:.3f} → {gpw_Ez}")

print("=" * 60)
print("[DONE] Field sweep completed")
