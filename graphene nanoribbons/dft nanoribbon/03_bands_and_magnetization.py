# 03_bands_and_magnetization.py
from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Riapri lo stato SCF ---
calc = GPAW('zgnr_scf.gpw')
atoms = calc.get_atoms()

pos = atoms.get_positions()
y = pos[:, 1]

print("=== INFO GEOMETRIA DAL GPW ===")
print("Cell:")
print(atoms.cell)
print("PBC:", atoms.pbc)
print("y min / max:", y.min(), y.max())
print("===============================\n")

# --- 2. Magnetizzazione atomica (per confronto con TB) ---
magmoms = calc.get_magnetic_moments()  # μ_B per atomo

# Ordina per coordinata y (dal bordo bottom al top)
order = np.argsort(y)
y_sorted = y[order]
mag_sorted = magmoms[order]
symbols_sorted = np.array(atoms.get_chemical_symbols())[order]

print("Magnetic moments per atom (sorted by y):")
for i, idx in enumerate(order):
    print(f"i={idx:3d}, symbol={atoms[idx].symbol:2s}, "
          f"y={y[idx]:7.3f} Å, m={magmoms[idx]:7.3f} μB")

print("\nTotal magnetic moment (sum over atoms):", magmoms.sum(), "μB\n")

# --- 3. Gruppo per "fila" trasversale: indice m discreto ---
# (semplice clusterizzazione per y usando una tolleranza)
tol = 0.2  # in Å, adatta se necessario
rows = []
current_row = [order[0]]
for a, b in zip(order[:-1], order[1:]):
    if abs(y[b] - y[a]) < tol:
        current_row.append(b)
    else:
        rows.append(current_row)
        current_row = [b]
rows.append(current_row)

print("Row-wise magnetization (approximate m index):")
for m_idx, row in enumerate(rows, start=1):
    m_mom = magmoms[row].sum()
    print(f"row m={m_idx:2d}, total moment = {m_mom:7.3f} μB, atoms = {list(row)}")
print()

# (opzionale) plot magnetizzazione per riga m
m_indices = np.arange(1, len(rows) + 1)
m_values = np.array([magmoms[row].sum() for row in rows])

plt.figure()
plt.bar(m_indices, m_values)
plt.xlabel("Row index m (across width)")
plt.ylabel("Total magnetic moment per row (μB)")
plt.title("ZGNR row-wise magnetization")
plt.tight_layout()
plt.savefig("zgnr_row_magnetization.png", dpi=300)
plt.close()
print("Saved 'zgnr_row_magnetization.png'\n")

# --- 4. Banda 1D lungo la direzione periodica ---

# Costruiamo un path Γ→X (assumendo x = direzione periodica)
# Se la periodicità è lungo un'altra direzione, cambia 'GX' di conseguenza.
path = atoms.cell.bandpath('GX', npoints=200)

# Calcolo non self-consistent delle bande lungo il path
bs_calc = calc.fixed_density(
    kpts=path.kpts,      # usa i k-points del BandPath
    symmetry='off',
    nbands=80,
    txt='zgnr_bands.txt'
)

# Ottieni BandStructure e usiamo direttamente il metodo plot di ASE
bs = bs_calc.band_structure(path)

# Plot automatico con ASE, riferito a bs.reference (tipicamente E_F = 0)
# Adatta emin/emax alla finestra che ti interessa (per le π-bands resta vicino a E_F)
emin, emax = -3.0, 3.0
bs.plot(filename='zgnr_bands.png', emin=emin, emax=emax, show=False)

print(f"Band structure saved to 'zgnr_bands.png' with energy window [{emin}, {emax}] eV.")
