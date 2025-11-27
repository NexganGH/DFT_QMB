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
points = {'G': [0, 0, 0],
          'K': [1/3, 1/3, 0],
          'M': [0, 1/2, 0]}

path = bandpath(
    path=points,
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
ef = calc_bs.get_fermi_level()

# -----------------------------
# 3. Plot band structure
# -----------------------------
plt.figure(figsize=(6, 5))

for n in range(nbands):
    plt.plot(x, eigs[:, n] - ef, lw=0.8)

for xc in xticks:
    plt.axvline(x=xc, color='0.8', lw=0.5)

plt.axhline(0.0, color='0.5', lw=0.7, ls='--')

plt.xticks(xticks, xlabels)
plt.ylabel('Energy (eV) relative to $E_F$')
plt.title('AB-stacked bilayer graphene (PBE, PW)')

plt.tight_layout()
plt.savefig('blg_bandstructure.png', dpi=300)
plt.show()
