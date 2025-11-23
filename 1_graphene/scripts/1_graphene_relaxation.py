from ase.build import graphene
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
from ase.io import Trajectory
from gpaw import GPAW, PW
import numpy as np

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
           logfile='relax.log',
           trajectory='graphene_relax.traj')

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

calc.write('graphene_relaxed.gpw')
