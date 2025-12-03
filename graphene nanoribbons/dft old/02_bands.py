from gpaw import GPAW
import matplotlib.pyplot as plt  # needed by bs.plot

# --- 1. Restart from spin-polarised ground state with full PBC ---

calc = GPAW('chain_spin_AF_pbc.gpw')   # adjust name if needed
atoms = calc.get_atoms()

print("Cell lengths (Å):", atoms.cell.lengths())
print("PBC:", atoms.pbc)  # should be [ True  True  True ]

# --- 2. Build a 1D band path (Γ → Z) along the chain direction ---

path = atoms.cell.bandpath('GZ', npoints=100)   # 10 k-points; increase later if you want smoother curves
print("Number of k-points along path:", len(path.kpts))

# --- 3. Non-selfconsistent band calculation with fixed density ---

bs_calc = calc.fixed_density(
    kpts=path.kpts,    # use the k-points along Γ–Z
    symmetry='off',    # no symmetry reduction
    nbands=10,         # 20 bands per k-point
    txt='chain_band.txt'
)

# --- 4. Build band-structure object (no path kwarg, no reference args) ---

bs = bs_calc.band_structure()
ef = bs_calc.get_fermi_level()
print("Fermi level (eV):", ef)

# --- 5. Plot (energies are absolute; EF is printed above) ---

# Simple plot: no energy shift, you just know EF from the printout
bs.plot(filename='chain_band.png', emax=3.0, emin=-3.0, show=True)
