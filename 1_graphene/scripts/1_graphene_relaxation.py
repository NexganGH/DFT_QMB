from ase.build import graphene
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
from ase.io import Trajectory
from gpaw import GPAW, PW
import numpy as np
import os

os.makedirs('../outputs', exist_ok=True)

# -----------------------------
# 1. Build graphene + vacuum
# -----------------------------
atoms = graphene()
atoms.center(vacuum=8.0, axis=2)

# -----------------------------
# 2. GPAW calculator
# -----------------------------
calc = GPAW(
    mode=PW(600),
    xc='PBE',
    kpts=(12, 12, 1),
    txt='graphene_relax.txt'
)
atoms.calc = calc

# -----------------------------
# 3. Relaxation + trajectory saving
# -----------------------------
mask = [True, True, False, True, False, False]
filter = ExpCellFilter(atoms, mask=mask)

traj = Trajectory('../outputs/graphene_relax.traj', 'w', atoms)

opt = BFGS(filter,
           logfile='../outputs/graphene_relax.log',
           trajectory='../outputs/graphene_relax.traj')

opt.run(fmax=0.02)

# -----------------------------
# 4. Extract structural observables
# -----------------------------
cell = atoms.cell
a = np.linalg.norm(cell[0, :2])
d_cc = atoms.get_distance(0, 1)
energy = atoms.get_potential_energy()

print("\nRelaxation completed.")
print(f"Optimised lattice constant a = {a:.4f} Å")
print(f"Optimised C–C bond length d_CC = {d_cc:.4f} Å")
print(f"Total energy per atom = {energy / len(atoms):.6f} eV")


# -----------------------------
# 5. Save important data to file
# -----------------------------
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

output_path = '../outputs/graphene_relax_data.txt'
with open(output_path, 'w') as f:
    f.write("Graphene Relaxation Results\n")
    f.write("Timestamp: {}\n\n".format(timestamp))

    f.write("=== Structural Properties ===\n")
    f.write(f"Lattice constant a: {a:.6f} Å\n")
    f.write(f"C–C bond length d_CC: {d_cc:.6f} Å\n")
    f.write("\n")

    f.write("=== Energetics ===\n")
    f.write(f"Total energy: {energy:.6f} eV\n")
    f.write(f"Energy per atom: {energy / len(atoms):.6f} eV/atom\n")
    f.write("\n")

    f.write("=== Computational Parameters ===\n")
    f.write("Exchange–correlation: PBE\n")
    f.write("Basis: Plane waves\n")
    f.write("Cutoff: 600 eV\n")
    f.write("k-points: 12 × 12 × 1\n")
    f.write("Vacuum spacing: 8 Å\n")
    f.write("Force threshold: 0.02 eV/Å\n")
    f.write("Cell relaxation mask: {}\n".format(mask))

print(f"\nSaved relaxation data to: {output_path}")


calc.write('graphene_relaxed.gpw')
