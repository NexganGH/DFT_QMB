from gpaw import GPAW
from ase.dft.kpoints import bandpath
import numpy as np

# -----------------------------------------------------
# 1. Load SCF ground-state
# -----------------------------------------------------
calc_scf = GPAW('../outputs/graphene_scf.gpw')

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


# -----------------------------
# 6. Save important band-structure data
# -----------------------------
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

outpath = '../outputs/graphene_bands_data.txt'

# Compute energy range (relative to Fermi level)
emin = eigs.min() - fermi
emax = eigs.max() - fermi

with open(outpath, 'w') as f:
    f.write("Graphene Band-Structure Calculation\n")
    f.write("Timestamp: {}\n\n".format(timestamp))

    f.write("=== k-Path ===\n")
    f.write("High-symmetry path: Γ → K → M → Γ\n")
    f.write(f"Total number of k-points: {N}\n\n")

    f.write("=== Energetics ===\n")
    f.write(f"Fermi level (from SCF): {fermi:.6f} eV\n")
    f.write(f"Minimum eigenvalue (shifted): {emin:.6f} eV\n")
    f.write(f"Maximum eigenvalue (shifted): {emax:.6f} eV\n\n")

    f.write("=== Band Information ===\n")
    f.write(f"Number of bands: {eigs.shape[1]}\n")
    f.write("Eigenvalues include both occupied and unoccupied states.\n\n")

    f.write("=== Computational Parameters ===\n")
    f.write("Exchange–correlation functional: PBE\n")
    f.write("Basis: Plane waves (PAW method)\n")
    f.write("Cutoff energy: 600 eV (inherited from SCF)\n")
    f.write("k-point path sampling: 200 points\n")
    f.write("Density used: fixed-density from SCF\n")
    f.write("Symmetry: off\n")

print(f"Saved band information to: {outpath}")

# -----------------------------------------------------
# 5. Save raw data to NPZ file
# -----------------------------------------------------
np.savez(
    '../outputs/graphene_bands_raw.npz',
    x=x,
    xticks=xticks,
    labels=np.array(['Γ', 'K', 'M', 'Γ']),
    eigs=eigs,
    fermi=fermi
)

print("Saved raw band-structure data to graphene_bands_raw.npz")
