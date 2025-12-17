import os
import sys
import csv
import numpy as np

from gpaw import GPAW

# ============================================================
# Configuration
# ============================================================
OUT_DIR = "../gpw"
FIELD_CSV = os.path.join(OUT_DIR, "field_sweep_results.csv")
OUT_CSV = os.path.join(OUT_DIR, "layer_polarisation_vs_field.csv")

LAYER_SEPARATION = 3.35  # Å
HALF_WINDOW = LAYER_SEPARATION / 2.0


# ============================================================
# Import and register external potentials
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from external_potentials import register_with_gpaw  # noqa: E402


# ============================================================
# Utilities
# ============================================================
def read_field_table(csv_path, out_dir):
    """
    Read Ez and gpw_path from field_sweep_results.csv
    """
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                Ez = float(row["Ez"])
            except Exception:
                continue

            gpw_path = row.get("gpw_path")
            if not gpw_path:
                gpw_path = os.path.join(out_dir, f"ab_gate_plane_A{Ez:.3f}.gpw")

            rows.append((Ez, gpw_path))

    return sorted(rows, key=lambda x: x[0])


def find_layer_centres(atoms):
    """
    Identify the z-centres of the two graphene layers
    from atomic positions.
    """
    z_positions = atoms.positions[:, 2]
    z_sorted = np.sort(z_positions)

    # Split atoms into two layers by z
    z_mid = 0.5 * (z_sorted[0] + z_sorted[-1])
    bottom_layer = z_positions[z_positions < z_mid]
    top_layer = z_positions[z_positions > z_mid]

    z_bot = bottom_layer.mean()
    z_top = top_layer.mean()

    return z_bot, z_top


def extract_layer_polarisation(gpw_path):
    """
    Compute n_top - n_bottom from planar-averaged density.
    """
    register_with_gpaw()
    calc = GPAW(gpw_path, txt=None)

    atoms = calc.atoms
    cell_z = atoms.cell[2, 2]

    # All-electron density
    rho = calc.get_all_electron_density(gridrefinement=2)

    # Planar average rho(z)
    rho_z = rho.mean(axis=(0, 1))
    nz = rho_z.size
    z_grid = np.linspace(0.0, cell_z, nz)

    # Layer centres
    z_bot, z_top = find_layer_centres(atoms)

    # Integration masks
    bot_mask = (z_grid > z_bot - HALF_WINDOW) & (z_grid < z_bot + HALF_WINDOW)
    top_mask = (z_grid > z_top - HALF_WINDOW) & (z_grid < z_top + HALF_WINDOW)

    n_bot = np.trapz(rho_z[bot_mask], z_grid[bot_mask])
    n_top = np.trapz(rho_z[top_mask], z_grid[top_mask])

    return n_top - n_bot


# ============================================================
# Main
# ============================================================
def main():
    rows = read_field_table(FIELD_CSV, OUT_DIR)

    results = []

    for Ez, gpw_path in rows:
        if not os.path.exists(gpw_path):
            print(f"[SKIP] Ez={Ez:.3f}: GPW not found")
            continue

        try:
            P = extract_layer_polarisation(gpw_path)
            results.append((Ez, P))
            print(f"[OK] Ez={Ez:.3f}  P={P:.6e}")
        except Exception as e:
            print(f"[FAIL] Ez={Ez:.3f}: {e}")

    # Write output CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ez", "layer_polarisation"])
        for Ez, P in results:
            writer.writerow([Ez, P])

    print(f"\nSaved layer polarisation to: {OUT_CSV}")


if __name__ == "__main__":
    main()
