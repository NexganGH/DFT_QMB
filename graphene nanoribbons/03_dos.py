from gpaw import GPAW
from ase.dft.dos import DOS
import matplotlib.pyplot as plt

# --- 1. Restart calculator ---

calc = GPAW('chain_spin_AF_pbc.gpw')

# --- 2. DOS object ---

dos = DOS(calc, width=0.1, npts=1001)  # width in eV, Gaussian smearing

energies = dos.get_energies()  # already relative to EF by default
dos_tot = dos.get_dos()        # total DOS
dos_up = dos.get_dos(spin=0)   # spin-up
dos_dn = dos.get_dos(spin=1)   # spin-down

# --- 3. Plot DOS ---

plt.figure()
plt.plot(energies, dos_tot, label='Total DOS')
plt.plot(energies, dos_up, '--', label='Spin up')
plt.plot(energies, dos_dn, '--', label='Spin down')
plt.axvline(0.0, color='k', linestyle=':')  # EF
plt.xlabel('E - E_F (eV)')
plt.ylabel('DOS (states/eV)')
plt.legend()
plt.tight_layout()
plt.savefig('chain_dos.png', dpi=200)
plt.show()
