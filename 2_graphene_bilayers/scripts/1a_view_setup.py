import numpy as np
from ase import Atoms
from ase.visualize import view
from gpaw import GPAW, FermiDirac, PW
from ase.dft.kpoints import get_special_points

a = 2.46
b = a / np.sqrt(3)
d = 3.35

grap_bilayer = Atoms('C' * 4,
                 positions=[(0,     0, 0),
                            (0,     b, 0),
                            (0,     b, d),
                            (0, 2 * b, d)],
                 cell=[[      a,       0,     0],
                       [0.5 * a, 1.5 * b,     0],
                       [      0,       0, 10 * a]
                       ],
                 pbc=True)

view(grap_bilayer.repeat((3, 3, 1)))