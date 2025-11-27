# blg_scf.py
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW

# -----------------------------
# 1. Build AB-stacked bilayer
# -----------------------------
a = 2.46  # in-plane lattice constant (Å)
c = 20.0  # cell height (Å) with vacuum
d = 3.35  # interlayer distance (Å), graphite-like

# Hexagonal cell vectors
a1 = np.array([a, 0.0, 0.0])
a2 = np.array([0.5 * a, np.sqrt(3) * 0.5 * a, 0.0])
a3 = np.array([0.0, 0.0, c])

cell = [a1, a2, a3]

# Helper: fractional -> Cartesian (keeping given z in Å)
def frac_to_cart(f1, f2, z_ang):
    return f1 * a1 + f2 * a2 + np.array([0.0, 0.0, z_ang])

# Place layers symmetrically around mid-plane
z_mid = c / 2.0
z1 = z_mid - d / 2.0  # bottom layer
z2 = z_mid + d / 2.0  # top layer

# Fractional in-plane coordinates (f1, f2) for AB stacking:
# Layer 1:
#   A1: (0, 0)
#   B1: (1/3, 2/3)
# Layer 2 (AB): A2 sits above B1, B2 is the other sublattice
#   A2: (1/3, 2/3)
#   B2: (2/3, 1/3)
frac_positions = [
    (0.0,       0.0,       z1),  # A1
    (1.0/3.0,   2.0/3.0,   z1),  # B1
    (1.0/3.0,   2.0/3.0,   z2),  # A2 (above B1)
    (2.0/3.0,   1.0/3.0,   z2),  # B2
]

cart_positions = [frac_to_cart(f1, f2, z) for (f1, f2, z) in frac_positions]

atoms = Atoms(
    symbols='C4',
    positions=cart_positions,
    cell=cell,
    pbc=[True, True, True],
)

# -----------------------------
# 2. Ground-state DFT (PBE PW)
# -----------------------------
calc = GPAW(
    mode=PW(600),  # 600 eV cutoff (you can reduce to 400 if needed)
    xc='PBE',
    kpts=(18, 18, 1),
    occupations={'name': 'fermi-dirac', 'width': 0.01},
    txt='blg_scf.txt',
)

atoms.calc = calc
E = atoms.get_potential_energy()
print(f'Total energy: {E:.6f} eV')

# Save full ground state for band-structure run
calc.write('blg_scf.gpw', mode='all')
