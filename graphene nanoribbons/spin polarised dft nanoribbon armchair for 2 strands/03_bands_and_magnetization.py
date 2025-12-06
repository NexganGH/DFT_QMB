# 03_bands_and_magnetization.py
from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Riapri lo stato SCF ---
calc = GPAW('zgnr_scf.gpw')
atoms = calc.get_atoms()

pos = atoms.get_positions()

print("=== INFO GEOMETRIA DAL GPW ===")
print("Cell:")
print(atoms.cell)
print("PBC:", atoms.pbc)
print("y min / max:", pos[:, 1].min(), pos[:, 1].max())
print("z min / max:", pos[:, 2].min(), pos[:, 2].max())
print("===============================\n")

# --- 2. Scegliamo la coordinata trasversale giusta (y o z) ---
span_y = pos[:, 1].max() - pos[:, 1].min()
span_z = pos[:, 2].max() - pos[:, 2].min()

if span_y > span_z:
    t = pos[:, 1]
    trans_label = 'y'
else:
    t = pos[:, 2]
    trans_label = 'z'

print(f"Using '{trans_label}' as transverse (width) coordinate.\n")

# --- 3. Magnetizzazione atomica (per confronto con TB) ---
magmoms = calc.get_magnetic_moments()  # μ_B per atomo

# Ordina per coordinata trasversale
order = np.argsort(t)

print("Magnetic moments per atom (sorted by transverse coord):")
for i, idx in enumerate(order):
    print(f"i={idx:3d}, symbol={atoms[idx].symbol:2s}, "
          f"{trans_label}={t[idx]:7.3f} Å, m={magmoms[idx]:7.3f} μB")

print("\nTotal magnetic moment (sum over atoms):", magmoms.sum(), "μB\n")

symbols = atoms.get_chemical_symbols()

# --- use only C atoms for the magnetization profile ---
isC = np.array([s == 'C' for s in symbols])
C_indices = np.where(isC)[0]

tC = t[C_indices]
posC = pos[C_indices]
magC = magmoms[C_indices]

# sort C atoms by transverse coordinate
orderC = np.argsort(tC)
C_sorted = C_indices[orderC]

tol = 0.2  # in Å, adjust if needed

rows = []
current_row = [C_sorted[0]]
for a, b in zip(C_sorted[:-1], C_sorted[1:]):
    if abs(t[b] - t[a]) < tol:
        current_row.append(b)
    else:
        rows.append(current_row)
        current_row = [b]
rows.append(current_row)

print("Number of C rows (should be Ny):", len(rows))

print("Row-wise magnetization using only C atoms, split into A/B-like (grouped rows):\n")

# For this ZGNR geometry: each "strand" m corresponds to 2 consecutive C-rows.
n_rows = len(rows)
if n_rows % 2 != 0:
    print("WARNING: number of C rows is odd; grouping by pairs may be wrong.")
n_strands = n_rows // 2

# Build groups of rows -> strands
rows_strands = []
for m in range(n_strands):
    group = rows[2*m] + rows[2*m + 1]
    rows_strands.append(group)

m_indices = []
mA = []
mB = []

x = pos[:, 0]  # periodic direction

for m_idx, group in enumerate(rows_strands, start=0):  # m = 0,1,...
    row = np.array(group, dtype=int)

    # sort atoms in this strand by x (periodic direction)
    order_x = np.argsort(x[row])
    row_sorted = row[order_x]

    # alternate A/B along x: even -> A-like, odd -> B-like
    A_atoms = row_sorted[0::2]
    B_atoms = row_sorted[1::2]

    mA_row = magmoms[A_atoms].mean()
    mB_row = magmoms[B_atoms].mean()

    m_indices.append(m_idx)
    mA.append(mA_row)
    mB.append(mB_row)

    print(f"m={m_idx:2d}: "
          f"A_atoms={A_atoms.tolist()}, mA={mA_row:7.3f} μB ; "
          f"B_atoms={B_atoms.tolist()}, mB={mB_row:7.3f} μB")

m_indices = np.array(m_indices)
mA = np.array(mA)
mB = np.array(mB)

# --- Plot in the same style as your MFT plot ---
plt.figure()
plt.plot(m_indices, mA, 'o-', label='mA')
plt.plot(m_indices, mB, 's-', label='mB')
plt.xlabel("Strand index m")
plt.ylabel("Magnetization m_m (μB)")
plt.legend()
plt.tight_layout()
plt.savefig("zgnr_magnetization_AB_vs_m.png", dpi=300)
plt.close()

print("\nSaved 'zgnr_magnetization_AB_vs_m.png' (A/B-resolved magnetization vs m)\n")


