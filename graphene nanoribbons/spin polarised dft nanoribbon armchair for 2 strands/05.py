# 05_pi_bands_extract.py
#
# Compute DFT bandstructure, select π-like bands near EF,
# plot them, and save them (k, E) to an .npz file for TB fitting.

from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Reload SCF calculation ---
calc = GPAW('zgnr_scf.gpw')
atoms = calc.get_atoms()

print("Original PBC from SCF:", atoms.pbc)

# ---- PATCH: make cell formally 3D periodic for band calculation ----
atoms.set_pbc((True, True, True))
calc.atoms.set_pbc((True, True, True))
print("PBC used for bands:", atoms.pbc)

# --- 2. Build band path Γ→X (same as in 03) ---
path = atoms.cell.bandpath('GX', npoints=200)

# --- 3. Non-selfconsistent band calculation ---
nbands = 40  # enough to include all π bands; increase if needed
bs_calc = calc.fixed_density(
    kpts=path.kpts,
    symmetry='off',
    nbands=nbands,
    txt='zgnr_bands_pi.txt'
)

bs = bs_calc.band_structure()

# energies: shape (nspins, nkpts, nbands)
energies = bs.energies
nspins, nkpts, nbands = energies.shape
print("nspins, nkpts, nbands =", nspins, nkpts, nbands)

# use Fermi level from this band calc
efermi = bs_calc.get_fermi_level()
print("Fermi level (eV):", efermi)

# energies relative to EF
E_rel = energies - efermi

# k-axis
x, X, labels = bs.get_labels()  # x: distances along path

# --- 4. Define π window ---
E_min_pi = -3.0  # eV
E_max_pi = +3.0  # eV
print(f"Selecting bands that enter [{E_min_pi}, {E_max_pi}] eV around EF.")

pi_band_indices_per_spin = []

for s in range(nspins):
    mask = (E_rel[s] > E_min_pi) & (E_rel[s] < E_max_pi)  # (nkpts, nbands)
    band_indices = [n for n in range(nbands) if mask[:, n].any()]
    pi_band_indices_per_spin.append(band_indices)
    print(f"Spin {s}: π-like band indices = {band_indices}")

# --- 5. Extract π-band energies for saving ---
E_pi_up = None
E_pi_dn = None
bands_up = np.array([], dtype=int)
bands_dn = np.array([], dtype=int)

if nspins == 1:
    bands_up = np.array(pi_band_indices_per_spin[0], dtype=int)
    if len(bands_up) > 0:
        E_pi_up = E_rel[0, :, bands_up]  # (nkpts, n_pi_up)
else:
    bands_up = np.array(pi_band_indices_per_spin[0], dtype=int)
    bands_dn = np.array(pi_band_indices_per_spin[1], dtype=int)
    if len(bands_up) > 0:
        E_pi_up = E_rel[0, :, bands_up]
    if len(bands_dn) > 0:
        E_pi_dn = E_rel[1, :, bands_dn]

# --- 6. Plot π bands ---
plt.figure(figsize=(5, 6))

colors = ['gold', 'royalblue']
labels_spin = ['spin up', 'spin down']

for s in range(nspins):
    band_list = pi_band_indices_per_spin[s]
    for n in band_list:
        plt.plot(x, E_rel[s, :, n],
                 color=colors[s % 2],
                 linewidth=1.0,
                 label=labels_spin[s] if n == band_list[0] else None)

plt.axhline(0.0, linestyle='--', linewidth=0.8, color='k')
plt.xticks(X, labels)
plt.ylim(E_min_pi, E_max_pi)
plt.xlabel(r'$k$ along $\Gamma$–X')
plt.ylabel(r'$E - E_F$ (eV)')
plt.title('ZGNR π-like bands (DFT)')
plt.legend()
plt.tight_layout()
plt.savefig('zgnr_pi_bands.png', dpi=300)
plt.close()

print("\nSaved π-only band plot as 'zgnr_pi_bands.png'")

# --- 7. Save bands to file for TB fitting ---
save_dict = {
    'k': x,               # (nkpts,)
    'efermi': efermi,
    'E_rel_all': E_rel,   # (nspins, nkpts, nbands)
    'bands_up': bands_up,
    'bands_dn': bands_dn,
}

if E_pi_up is not None:
    save_dict['E_pi_up'] = E_pi_up   # (nkpts, n_pi_up)
if E_pi_dn is not None:
    save_dict['E_pi_dn'] = E_pi_dn   # (nkpts, n_pi_dn)

import numpy as np
np.savez('zgnr_pi_bands.npz', **save_dict)
print("Saved π-band data to 'zgnr_pi_bands.npz'")
