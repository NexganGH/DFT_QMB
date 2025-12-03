from ase.io import read
from ase.visualize import view
import numpy as np

init = read('chain_initial.traj')
relaxed = read('chain_relaxed.traj')

# Shift relaxed structure in x so they don't overlap
relaxed_shifted = relaxed.copy()
relaxed_shifted.translate([10.0, 0.0, 0.0])

# Colors
init_colors = np.array([[0.2, 0.2, 1.0]] * len(init))
relaxed_colors = np.array([[1.0, 0.2, 0.2]] * len(relaxed_shifted))

init.set_array('color', init_colors)
relaxed_shifted.set_array('color', relaxed_colors)

view([init, relaxed_shifted])
