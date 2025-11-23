from gpaw import GPAW, mpi
from ase.dft.kpoints import bandpath
from ase.parallel import parprint
import numpy as np

# -----------------------------------------------------
# 1. Load SCF ground-state from old file
# -----------------------------------------------------
# This file must come from your previous SCF run
calc_scf = GPAW('../outputs/graphene_scf.gpw')
atoms = calc_scf.atoms

parprint("Loaded SCF from graphene_scf.gpw")

# -----------------------------------------------------
# 2. Build band path (Γ–K–M–Γ)
# -----------------------------------------------------
path = bandpath(
    ['G', 'K', 'M', 'G'],
    cell=atoms.cell,
    npoints=200
)

kpts = path.kpts
x, xticks, _ = path.get_linear_kpoint_axis()
parprint(f"Generated {len(kpts)} k-points for band path")

# -----------------------------------------------------
# 3. Non-SCF band calculation with native GPAW parallelization
# -----------------------------------------------------
calc = calc_scf.fixed_density(
    kpts=kpts,
    symmetry='off',
    parallel={'kpt': True},      # let GPAW parallelize over k-points
    txt='graphene_bands.txt'
)

# This does the heavy Hamiltonian setup + diagonalization
calc.get_potential_energy()

# -----------------------------------------------------
# 4. Collect eigenvalues (GPAW handles MPI internally)
# -----------------------------------------------------
N = len(kpts)
eigs = np.array([calc.get_eigenvalues(kpt=i) for i in range(N)])
fermi = calc.get_fermi_level()

# -----------------------------------------------------
# 5. Save data only on master rank
# -----------------------------------------------------
if mpi.rank == 0:
    np.savez(
        '../outputs/graphene_bands_raw.npz',
        eigs=eigs,
        fermi=fermi,
        x=x,
        xticks=xticks
    )
    parprint("Saved: graphene_bands_raw.npz")
    parprint("Band structure calculation complete.")
