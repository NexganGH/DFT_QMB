from gpaw import GPAW
import numpy as np

calc = GPAW("zgnr4_nonmag_relaxed.gpw")
atoms = calc.get_atoms()

pos = atoms.get_positions()
cell = atoms.cell

print("Cell vectors (Å):")
print(cell)

# width = spread across non-periodic direction (y if zigzag along x)
span = pos.max(axis=0) - pos.min(axis=0)
print("Span (Å) along x, y, z:", span)

# distance between outermost edge carbons across the width
# assuming width is along y:
y = pos[:, 1]
width = y.max() - y.min()
print("Edge-to-edge width (Å):", width)
