# compute_pdos_pi_sigma.py
import os
import numpy as np
from gpaw import GPAW
from gpaw.dos import DOSCalculator

# ------------------------------------------------
# Paths and setup
# ------------------------------------------------
outdir = '../outputs/'
os.makedirs(outdir, exist_ok=True)

gpw_file = os.path.join(outdir, 'graphene_scf.gpw')

# Load calculator and build DOSCalculator
calc = GPAW(gpw_file, txt=None)
doscalc = DOSCalculator.from_calculator(calc)

# Energy grid in eV (DOSCalculator already shifts by E_F)
emin, emax = -10.0, 10.0
npts = 2001
energies = np.linspace(emin, emax, npts)

# Broadening (eV)
width = 0.2

# We’ll average over the two C atoms in the primitive cell
atom_indices = [0, 1]

def avg_pdos_over_atoms(l, m=None):
    """Average PDOS over selected atoms for given (l, m)."""
    pdos_sum = np.zeros_like(energies)
    for a in atom_indices:
        pdos = doscalc.raw_pdos(
            energies,
            a=a,
            l=l,
            m=m,
            spin=None,   # total (sum over spins)
            width=width
        )
        pdos_sum += pdos
    return pdos_sum / len(atom_indices)

# -----------------------------
# s, total p, p_z, sigma = p_x + p_y
# -----------------------------

# s channel (l = 0)
pdos_s = avg_pdos_over_atoms(l=0, m=None)

# total p (p_x + p_y + p_z), l = 1, sum over m
pdos_p_total = avg_pdos_over_atoms(l=1, m=None)

# p_z only:
# For p orbitals: m = 0,1,2 -> y, z, x  (from GPAW docs)
pdos_pz = avg_pdos_over_atoms(l=1, m=1)

# Sigma = p_x + p_y = total p - p_z
pdos_sigma = pdos_p_total - pdos_pz

# Numerical noise can make tiny negatives; clip them
pdos_sigma = np.clip(pdos_sigma, 0.0, None)
pdos_pz = np.clip(pdos_pz, 0.0, None)

# ------------------------------------------------
# Save to files
# ------------------------------------------------
np.savetxt(
    os.path.join(outdir, 'graphene_pdos_s.dat'),
    np.column_stack((energies, pdos_s)),
    header='Energy - EF (eV)   s-PDOS (avg over atoms)'
)

np.savetxt(
    os.path.join(outdir, 'graphene_pdos_p_total.dat'),
    np.column_stack((energies, pdos_p_total)),
    header='Energy - EF (eV)   total p-PDOS (px+py+pz, avg over atoms)'
)

np.savetxt(
    os.path.join(outdir, 'graphene_pdos_pi.dat'),
    np.column_stack((energies, pdos_pz)),
    header='Energy - EF (eV)   pi-PDOS (p_z, avg over atoms)'
)

np.savetxt(
    os.path.join(outdir, 'graphene_pdos_sigma.dat'),
    np.column_stack((energies, pdos_sigma)),
    header='Energy - EF (eV)   sigma-PDOS (p_x + p_y, avg over atoms)'
)

print("Saved PDOS files in", outdir)
