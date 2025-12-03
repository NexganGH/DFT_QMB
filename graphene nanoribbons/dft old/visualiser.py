from ase.io import read
from ase.visualize import view

# Read the relaxed primitive cell
atoms = read('chain_relaxed.traj')   # or chain_initial.traj

# Repeat the cell: (nx, ny, nz)
# For a 1D chain along z: repeat many times in z only
supercell = atoms.repeat((1, 1, 10))   # 10 copies along the chain direction

# Open interactive window
view(supercell)

