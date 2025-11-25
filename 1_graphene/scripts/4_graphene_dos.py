from gpaw import GPAW, restart
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------
# Ensure output directory exists
# ----------------------------------------------------
outdir = '../outputs/'
os.makedirs(outdir, exist_ok=True)

# ----------------------------------------------------
# 1. Load SCF calculation
# ----------------------------------------------------
atoms, calc = restart(outdir + 'graphene_scf.gpw')
e_f = calc.get_fermi_level()

# ----------------------------------------------------
# 2. Compute DOS
# ----------------------------------------------------
# npts — resolution
# width — Gaussian broadening in eV
e, dos = calc.get_dos(spin=0, npts=2001, width=0.2)




# ----------------------------------------------------
# 3. Save DOS data (optional)
# ----------------------------------------------------
np.savetxt(
    outdir + 'graphene_dos.dat',
    np.column_stack((e - e_f, dos)),
    header='Energy - EF (eV)   DOS (states/eV)'
)


