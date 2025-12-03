# 02_scf_zgnr.py
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
import numpy as np

# --- 1. Carica la struttura ---
atoms = read('zgnr_Ny8_M1_ribbon.traj')

# --- DEBUG geometria ---
pos = atoms.get_positions()
print("=== GEOMETRIA ZGNR (SCF) ===")
print("Cell:")
print(atoms.cell)
print("PBC:", atoms.pbc)
print("y min / max:", pos[:, 1].min(), pos[:, 1].max())
print("z min / max:", pos[:, 2].min(), pos[:, 2].max())
print("First 10 positions (x, y, z):")
print(pos[:10])
print("=============================\n")

# --- 2. Scegliamo automaticamente l'asse trasversale (per l'indice m e i bordi) ---
span_y = pos[:, 1].max() - pos[:, 1].min()
span_z = pos[:, 2].max() - pos[:, 2].min()

if span_y > span_z:
    trans_coord = pos[:, 1]
    trans_label = 'y'
else:
    trans_coord = pos[:, 2]
    trans_label = 'z'

print(f"Using '{trans_label}' as transverse (width) coordinate.\n")

# --- 3. Inizializza momento magnetico (edge AF) ---
t = trans_coord
t_min, t_max = t.min(), t.max()
t_mid = 0.5 * (t_min + t_max)

magmoms_init = []
for ti, symbol in zip(t, atoms.get_chemical_symbols()):
    if symbol == 'C':
        if ti < t_mid:
            magmoms_init.append(+0.3)  # bordo "bottom"
        else:
            magmoms_init.append(-0.3)  # bordo "top"
    else:
        magmoms_init.append(0.0)      # H non magnetico

atoms.set_initial_magnetic_moments(magmoms_init)

print("Initial magmoms (first 20):", magmoms_init[:20])
print("Initial total magmom:", sum(magmoms_init), "\n")

# --- 4. Definisci il calcolatore GPAW ---
Nk = 40

calc = GPAW(mode=PW(400),          # aumenta a 300-400 eV per risultati seri
            xc='PBE',
            kpts=(Nk, 1, 1),
            occupations=FermiDirac(0.001),
            spinpol=True,
            symmetry='off',
            txt='zgnr_scf.txt')

atoms.calc = calc

# --- 5. Run SCF ---
print("Starting SCF calculation...")
E_tot = atoms.get_potential_energy()
print("SCF finished.")
print("Total energy (eV):", E_tot)

# --- 6. Magnetizzazione finale ---
final_magmoms = calc.get_magnetic_moments()
print("\n=== MAGNETIZZAZIONE FINALE ===")
print("Final magmoms (first 20):", final_magmoms[:20])
print("Total magnetic moment:", final_magmoms.sum())
print("==============================\n")

order = np.argsort(t)
print("Magnetic moments per atom (sorted by transverse coord):")
for i, idx in enumerate(order):
    print(f"i={idx:3d}, symbol={atoms[idx].symbol:2s}, "
          f"{trans_label}={t[idx]:7.3f} Å, m={final_magmoms[idx]:7.3f} μB")

calc.write('zgnr_scf.gpw', mode='all')
print("\nSCF state written to 'zgnr_scf.gpw'")
