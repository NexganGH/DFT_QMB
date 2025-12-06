# 06_replot_pi_window.py
#
# Re-plot π-like bands from saved data, with adjustable energy window,
# WITHOUT rerunning the DFT band calculation.

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load saved band data ---
data = np.load('zgnr_pi_bands.npz')

k = data['k']                  # (nkpts,)
E_rel_all = data['E_rel_all']  # (nspins, nkpts, nbands)
nspins, nkpts, nbands = E_rel_all.shape

print("Loaded E_rel_all with shape:", E_rel_all.shape)

# --- 2. Choose energy window around EF (adjust freely!) ---
E_min = -6.0   # eV
E_max =  6.0   # eV
print(f"Plotting bands that enter [{E_min}, {E_max}] eV")

# --- 3. Choose bands in that window ---
pi_band_indices_per_spin = []

for s in range(nspins):
    mask = (E_rel_all[s] > E_min) & (E_rel_all[s] < E_max)
    band_indices = [n for n in range(nbands) if mask[:, n].any()]
    pi_band_indices_per_spin.append(band_indices)
    print(f"Spin {s}: bands in window =", band_indices)

# --- 4. Plot ---
x = k  # already the distance along Γ–X (from 05)

plt.figure(figsize=(5, 6))
colors = ['gold', 'royalblue']
labels_spin = ['spin up', 'spin down']

for s in range(nspins):
    band_list = pi_band_indices_per_spin[s]
    for n in band_list:
        plt.plot(x, E_rel_all[s, :, n],
                 color=colors[s % 2],
                 linewidth=1.0,
                 label=labels_spin[s] if n == band_list[0] else None)

plt.axhline(0.0, linestyle='--', linewidth=0.8, color='k')
plt.xlabel(r'$k$ along $\Gamma$–X')
plt.ylabel(r'$E - E_F$ (eV)')
plt.ylim(E_min, E_max)
plt.title('ZGNR bands in chosen window')
plt.legend()
plt.tight_layout()
plt.savefig('zgnr_pi_bands_window.png', dpi=300)
plt.close()

print("Saved 'zgnr_pi_bands_window.png'")
