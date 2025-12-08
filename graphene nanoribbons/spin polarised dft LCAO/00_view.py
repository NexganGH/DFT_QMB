# 00_view_penguin.py
#
# Launch the ASE GUI (the viewer with the penguin logo)
# to inspect the nanoribbon interactively.

import config_zgnr_LCAO as cfg
from ase.io import read
from ase.visualize import view

# Load the geometry built by 01_zgnr_geometry.py
atoms = read(cfg.geom_traj)

print("Opening ASE GUI (penguin viewer)...")
print("Rotate with mouse. Press 'a' to show axes. Press 'c' for cell.")

view(atoms)
