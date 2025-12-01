import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load saved band data ---
data = np.load('chain_bands_raw.npz')

energies = data['energies']   # shape (nspin, nk, nb)
kdist    = data['kdist']      # x-axis (distance along Γ–Z)
ef       = float(data['ef'])

# Shift energies so EF = 0 (optional but convenient)
energies -= ef

nspin, nk, nb = energies.shape
print(f"Loaded: nspin={nspin}, nk={nk}, nb={nb}, EF=0 in this plot")

# --- 2. Choose energy window to display ---
emin = -12.0   # change these two numbers whenever you want
emax =  3.0

# --- 3. Plot ---
plt.figure(figsize=(6,4))

for s in range(nspin):
    for b in range(nb):
        E = energies[s, :, b]
        # Only plot if the band crosses the window
        if np.any((E >= emin) & (E <= emax)):
            style = 'b-' if s == 1 else 'g-'
            plt.plot(kdist, E, style, linewidth=0.8)

plt.axhline(0.0, color='k', linestyle='--', linewidth=0.7)  # EF
plt.ylim(emin, emax)
plt.xlabel('k-path (Γ → Z)')
plt.ylabel('Energy - E_F (eV)')
plt.title('Band structure (custom energy window)')
plt.tight_layout()
plt.show()
