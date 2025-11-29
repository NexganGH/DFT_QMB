# blg_bands.py
import numpy as np
import matplotlib.pyplot as plt
from ase.dft.kpoints import bandpath
from gpaw import GPAW

# -----------------------------
# 1. Load SCF and build path
# -----------------------------
# Load SCF calc (density, potential)
calc = GPAW('blg_scf.gpw', txt='blg_bands.txt')

atoms = calc.atoms
cell = atoms.cell

# High-symmetry path for hexagonal lattice:
#   Γ (0, 0, 0)
#   K (1/3, 1/3, 0)
#   M (0, 1/2, 0)
# points = {'G': [0, 0, 0],
#           'K': [1/3, 1/3, 0],
#           'M': [0, 1/2, 0],
#           'G': [0, 0, 0]}

path = bandpath(
    path=[['G', 'K', 'M', 'G']],
    cell=cell,
    npoints=200,
)

kpts = path.kpts
x, xticks, xlabels = path.get_linear_kpoint_axis()


# -----------------------------
# 2. Non-selfconsistent band calc
# -----------------------------
# Reuse density; turn off symmetry to keep all k-points
calc_bs = calc.fixed_density(
    kpts=kpts,
    symmetry='off',
    txt='blg_bands_fixed_density.txt',
)

nbands = calc_bs.get_number_of_bands()
eigs = np.zeros((len(kpts), nbands))

for i_k, kpt in enumerate(kpts):
    eigs[i_k, :] = calc_bs.get_eigenvalues(kpt=i_k)

# Fermi level
fermi = calc_bs.get_fermi_level()

# -----------------------------
# 3. Plot band structure
# -----------------------------


# -----------------------------
# 6. Save important band-structure data
# -----------------------------
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# outpath = '../outputs/graphene_bands_data.txt'
#
# # Compute energy range (relative to Fermi level)
# emin = eigs.min() - fermi
# emax = eigs.max() - fermi
#
# with open(outpath, 'w') as f:
#     f.write("Graphene Band-Structure Calculation\n")
#     f.write("Timestamp: {}\n\n".format(timestamp))
#
#     f.write("=== k-Path ===\n")
#     f.write("High-symmetry path: Γ → K → M → Γ\n")
#     f.write(f"Total number of k-points: {N}\n\n")
#
#     f.write("=== Energetics ===\n")
#     f.write(f"Fermi level (from SCF): {fermi:.6f} eV\n")
#     f.write(f"Minimum eigenvalue (shifted): {emin:.6f} eV\n")
#     f.write(f"Maximum eigenvalue (shifted): {emax:.6f} eV\n\n")
#
#     f.write("=== Band Information ===\n")
#     f.write(f"Number of bands: {eigs.shape[1]}\n")
#     f.write("Eigenvalues include both occupied and unoccupied states.\n\n")
#
#     f.write("=== Computational Parameters ===\n")
#     f.write("Exchange–correlation functional: PBE\n")
#     f.write("Basis: Plane waves (PAW method)\n")
#     f.write("Cutoff energy: 600 eV (inherited from SCF)\n")
#     f.write("k-point path sampling: 200 points\n")
#     f.write("Density used: fixed-density from SCF\n")
#     f.write("Symmetry: off\n")
#
# print(f"Saved band information to: {outpath}")

# -----------------------------------------------------
# 5. Save raw data to NPZ file
# -----------------------------------------------------
np.savez(
    '../outputs/bilayer_bands_raw.npz',
    x=x,
    xticks=xticks,
    labels=np.array(['Γ', 'K', 'M', 'Γ']),
    eigs=eigs,
    fermi=fermi
)

print("Saved raw band-structure data to bilayer_bands_raw.npz")

#plt.figure(figsize=(6, 5))
#
# for n in range(nbands):
#     plt.plot(x, eigs[:, n] - ef, lw=0.8)
#
# for xc in xticks:
#     plt.axvline(x=xc, color='0.8', lw=0.5)
#
# plt.axhline(0.0, color='0.5', lw=0.7, ls='--')
#
# plt.xticks(xticks, xlabels)
# plt.ylabel('Energy (eV) relative to $E_F$')
# plt.title('AB-stacked bilayer graphene (PBE, PW)')
#
# plt.tight_layout()
# plt.savefig('blg_bandstructure.png', dpi=300)
# plt.show()
