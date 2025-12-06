# 06_view_structure.py

from ase.visualize import view
from gpaw import GPAW
from config_zgnr import LABEL_BASE

# Load relaxed structure
gpw_file = "zgnr4_nonmag_relaxed.gpw"
calc = GPAW(gpw_file)
atoms = calc.get_atoms()

print("Loaded:", gpw_file)
print("Opening ASE viewer...")

view(atoms)  # interactive visualization
