from gpaw import GPAW
from ase.dft.kpoints import bandpath
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load SCF ground-state
# -----------------------------------------------------
calc_scf = GPAW('graphene_lda_scf.gpw')

# -----------------------------------------------------
# 2. Build high-symmetry k-path
# -----------------------------------------------------
path = bandpath(
    ['G', 'K', 'M', 'G'],
    cell=calc_scf.atoms.cell,
    npoints=200
)

path.plot()

kpts = path.kpts  # list of k-points on the path

# -----------------------------------------------------
# 3. Create non-SCF calculator (new API)
# -----------------------------------------------------
calc = calc_scf.fixed_density(
    kpts=kpts,
    symmetry='off',
    txt='graphene_bands.txt'
)

# Trigger wavefunction evaluation
calc.get_potential_energy()

# -----------------------------------------------------
# 4. Extract eigenvalues
# -----------------------------------------------------
eigs = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(kpts))])
fermi = calc.get_fermi_level()

# -----------------------------------------------------
# 5. Plot band structure
# -----------------------------------------------------
# x = path.distance
# xticks = path.x
x, xticks, _ = path.get_linear_kpoint_axis()

plt.figure(figsize=(7, 5))
plt.plot(x, eigs - fermi, 'k-', lw=1)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)

# vertical lines at special points
for xpos in xticks:
    plt.axvline(xpos, color='gray', linewidth=0.5)

plt.xticks(xticks, ['Γ', 'K', 'M', 'Γ'])
plt.ylabel('Energy (eV)')
plt.title('Graphene Band Structure (LDA/PBE)')
plt.tight_layout()
plt.savefig('graphene_bandstructure.png', dpi=200)
plt.show()
