from gpaw import GPAW, restart
import numpy as np

# --- load your calc from the gpw file ---
atoms, calc = restart('DFT.gpw')  # or GPAW('bilayer.gpw')

# --- get band structure along a path if you haven't already ---
# If you already did a band_structure() before and saved it, you can skip this
from ase.dft.kpoints import get_bandpath
from ase.dft.band_structure import BandStructure
points = {'M': [0.5, 0.0, 0.0],
          'K': [1/3, 1/3, 0.0],
          'G': [0.0, 0.0, 0.0]}
path = ['M', 'K', 'G']
path = get_bandpath(path, atoms.cell, npoints=20)#kpts, x, X = get_bandpath(path, atoms.cell, npoints=200)
kpts = path.kpts
x, X, labels = path.get_linear_kpoint_axis()


#atoms, calc = restart('DFT.gpw')

calc_bs = calc.fixed_density(
    kpts=kpts,
    symmetry='off',
    txt='blg_bands_fixed_density.txt',
)
from ase.dft.kpoints import kpoint_convert

bs = calc_bs.band_structure()
kpts_cart = kpoint_convert(atoms.cell, skpts_kc=kpts)#bs.get_kpoints(cartesian=True)
print(kpts_cart)
# Energies and Fermi level
E = bs.energies[0]        # shape: (Nk, Nbands)
EF = calc_bs.get_fermi_level()

# --- find π bands: 4 bands closest to EF ---
# (2 π bands per layer → 4 for bilayer)
dist = np.abs(E.mean(axis=0) - EF)   # distance of each band from EF
idx_pi = np.argsort(dist)[:4]        # indices of π bands

print("π-band indices:", idx_pi)

# Extract π-band energies (shifted so EF = 0)
E_pi = E[:, idx_pi] - EF             # shape: (Nk, 4)

# Optionally save for TB fitting
np.savez("pi_bands_from_gpaw.npz",
         kpts=kpts_cart[:, :2],  # kx, ky
         energies=E_pi,
         band_indices=idx_pi,
         EF=EF)
