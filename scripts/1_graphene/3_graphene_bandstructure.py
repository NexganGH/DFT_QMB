from gpaw import GPAW
from ase.dft.kpoints import bandpath
import numpy as np

# -----------------------------------------------------
# 1. Load SCF ground-state
# -----------------------------------------------------
calc_scf = GPAW('graphene_scf.gpw')

# -----------------------------------------------------
# 2. Build high-symmetry k-path
# -----------------------------------------------------
path = bandpath(
    ['G', 'K', 'M', 'G'],
    cell=calc_scf.atoms.cell,
    npoints=200
)

kpts = path.kpts
x, xticks, xlabels = path.get_linear_kpoint_axis()

# -----------------------------------------------------
# 3. Band calculation via fixed density
# -----------------------------------------------------
calc = calc_scf.fixed_density(
    kpts=kpts,
    symmetry='off',
    txt='graphene_bands.txt'
)

calc.get_potential_energy()   # trigger diagonalization

# -----------------------------------------------------
# 4. Extract eigenvalues
# -----------------------------------------------------
# -----------------------------------------------------
# 4. Extract eigenvalues with progress
# -----------------------------------------------------
eigs = []
N = len(kpts)

for i in range(N):
    print(f"[{i+1}/{N}] Diagonalizing at k-point {i} ...")
    eigs.append(calc.get_eigenvalues(kpt=i))

eigs = np.array(eigs)
fermi = calc.get_fermi_level()


# -----------------------------------------------------
# 5. Save raw data to NPZ file
# -----------------------------------------------------
np.savez(
    'graphene_bands_raw.npz',
    x=x,
    xticks=xticks,
    labels=np.array(['Γ', 'K', 'M', 'Γ']),
    eigs=eigs,
    fermi=fermi
)

print("Saved raw band-structure data to graphene_bands_raw.npz")
