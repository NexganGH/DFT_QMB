# 04_pdos_pz.py
from gpaw import GPAW
from gpaw.dos import DOSCalculator
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load SCF calculation ---
calc = GPAW('zgnr_scf.gpw')
atoms = calc.get_atoms()
symbols = atoms.get_chemical_symbols()

efermi = calc.get_fermi_level()
print("Fermi level (eV):", efermi)

# --- 2. Select carbon atoms ---
c_indices = [i for i, s in enumerate(symbols) if s == 'C']
print("C atom indices:", c_indices)
print("Number of C atoms:", len(c_indices))

# --- 3. Set up DOS calculator ---
doscalc = DOSCalculator.from_calculator(calc)

# Energy grid around EF (adjust range to compare with TB)
emin = -3.0  # eV relative to EF
emax =  3.0
npts = 2001
energies = np.linspace(efermi + emin, efermi + emax, npts)  # absolute energies
E_shift = energies - efermi                                 # E - EF

width = 0.2  # Gaussian broadening in eV

# --- 4. Total DOS ---
DOS_tot = doscalc.raw_dos(energies, width=width)

# --- 5. C-p (l=1) projected DOS ---
DOS_p = np.zeros_like(energies)
for a in c_indices:
    DOS_p += doscalc.raw_pdos(energies, a=a, l=1, width=width)

# --- 6. Plot ---
plt.figure(figsize=(6, 5))
plt.plot(E_shift, DOS_tot, label="Total DOS")
plt.plot(E_shift, DOS_p, '--', label="C p-like DOS (≈ π)")

plt.axvline(0.0, color='k', linestyle=':', linewidth=0.8)
plt.xlabel(r"$E - E_F$ (eV)")
plt.ylabel("DOS (arb. units)")
plt.title("ZGNR DOS and C p-like DOS")
plt.legend()
plt.tight_layout()
plt.savefig("zgnr_p_like_pdos_DOSCalculator.png", dpi=300)
plt.close()

print("\nSaved 'zgnr_p_like_pdos_DOSCalculator.png'")
