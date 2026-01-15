import os
import sys
import csv
import numpy as np

from gpaw import GPAW

#=============================================================
# Here we extract the polarization to later use for fitting.
#============================================================


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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts/tight_binding"))
sys.path.insert(0, PROJECT_ROOT)

from external_potentials import register_with_gpaw  # noqa: E402


# ============================================================
# Geometry / density utilities
# ============================================================
def cell_area(atoms):
    a1 = atoms.cell[0]
    a2 = atoms.cell[1]
    return np.linalg.norm(np.cross(a1, a2))


def planar_average_density(calc, gridrefinement=2):
    """
    Returns z-grid and planar-averaged all-electron density rho(z)
    rho in e/Å^3, z in Å
    """
    rho = calc.get_all_electron_density(gridrefinement=gridrefinement)
    rho_z = rho.mean(axis=(0, 1))
    Lz = float(calc.atoms.cell[2, 2])
    z = np.linspace(0.0, Lz, rho_z.size)
    return z, rho_z, Lz


def layer_centres(atoms):
    """
    Determine layer centres from atomic z-positions.
    """
    zpos = atoms.positions[:, 2]
    z_sorted = np.sort(zpos)
    z_mid = 0.5 * (z_sorted[0] + z_sorted[-1])

    z_bot = zpos[zpos < z_mid].mean()
    z_top = zpos[zpos > z_mid].mean()

    return float(z_bot), float(z_top)


def periodic_mask(z, z0, half_width, Lz):
    """
    Periodic distance mask around z0 with minimal-image convention.
    """
    dz = (z - z0 + 0.5 * Lz) % Lz - 0.5 * Lz
    return np.abs(dz) <= half_width


def integrate_layer_charge(z, drho_z, z0, half_width, Lz, area):
    """
    Integrate induced charge around one layer.
    Returns electrons per unit cell.
    """
    mask = periodic_mask(z, z0, half_width, Lz)
    # drho_z: e/Å^3 → integrate dz → e/Å^2 → multiply by area
    return area * np.trapz(drho_z[mask], z[mask])


# ============================================================
# I/O
# ============================================================
def read_field_table(csv_path, out_dir):
    """
    Returns sorted list of (Ez, gpw_path)
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


# ============================================================
# Main extraction
# ============================================================
def main():
    rows = read_field_table(FIELD_CSV, OUT_DIR)

    # --------------------------------------------------------
    # Locate Ez = 0 reference
    # --------------------------------------------------------
    ref = None
    for Ez, gpw in rows:
        if abs(Ez) < 1e-12:
            ref = gpw
            break

    if ref is None:
        raise RuntimeError("No Ez = 0 reference found in field sweep.")

    register_with_gpaw()
    calc0 = GPAW(ref, txt=None)

    z0, rho0, Lz0 = planar_average_density(calc0)
    atoms0 = calc0.atoms
    area = cell_area(atoms0)
    z_bot, z_top = layer_centres(atoms0)

    # --------------------------------------------------------
    # Loop over fields
    # --------------------------------------------------------
    results = []

    for Ez, gpw in rows:
        if Ez > 0.02: continue
        if not os.path.exists(gpw):
            print(f"[SKIP] Ez={Ez:.3f}: GPW not found")
            continue

        register_with_gpaw()
        calc = GPAW(gpw, txt=None)

        z, rho, Lz = planar_average_density(calc)

        if rho.size != rho0.size or abs(Lz - Lz0) > 1e-6:
            raise RuntimeError(f"Inconsistent grid/cell for Ez={Ez:.3f}")

        drho = rho - rho0

        Q_top = integrate_layer_charge(
            z, drho, z_top, HALF_WINDOW, Lz, area
        )
        Q_bot = integrate_layer_charge(
            z, drho, z_bot, HALF_WINDOW, Lz, area
        )

        P = Q_top - Q_bot
        results.append((Ez, P))

        print(f"[OK] Ez={Ez:.3f}  P_ind={P:.6e}")

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ez", "layer_polarisation_electrons"])
        for Ez, P in results:
            writer.writerow([Ez, P])

    print(f"\nSaved induced polarisation to: {OUT_CSV}")


if __name__ == "__main__":
    main()
