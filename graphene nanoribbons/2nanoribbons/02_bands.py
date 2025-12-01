from gpaw import GPAW
import numpy as np

# --- 1. Restart from spin-polarised ground state with full PBC ---
calc = GPAW('chain_spin_AF_pbc.gpw')
atoms = calc.get_atoms()

print("Cell lengths (Å):", atoms.cell.lengths())
print("PBC:", atoms.pbc)

# --- 2. Build a 1D band path (Γ → Z) along the chain direction ---
path = atoms.cell.bandpath('GZ', npoints=10)
kpts = path.kpts
print("Number of k-points along path:", len(kpts))

# --- 3. Non-selfconsistent band calculation with fixed density ---
bs_calc = calc.fixed_density(
    kpts=path,          # 💡 pass the BandPath object, not just kpts
    symmetry='off',
    nbands=10,
    txt='chain_band.txt'
)

# --- 4. Build band-structure object and extract data ---
bs = bs_calc.band_structure()
ef = bs_calc.get_fermi_level()
print("Fermi level (eV):", ef)

energies = bs.energies                # (nspins, nkpts, nbands)

# This gives the "distance along path" + tick positions + labels:
kdist, xticks, xlabels = bs.get_labels()

# --- 5. Save everything to a numpy file ---
np.savez(
    'chain_bands_raw.npz',
    energies=energies,
    kdist=kdist,
    ef=ef,
    kpts=kpts,
    xticks=xticks,
    xlabels=xlabels
)

print("Saved band data to chain_bands_raw.npz")
