# 01_build_zgnr.py
from ase.build import graphene_nanoribbon
from ase.io import write
import numpy as np

# --- Parametri "geometrici" ---
Ny = 5   # larghezza in linee zigzag (questo è il tuo "N_y")
M = 1    # numero di celle lungo la direzione periodica (lungo il ribbon)

# ATTENZIONE: in ASE l'ordine è (length, width),
# quindi: primo argomento = M (lungo il ribbon), secondo = Ny (larghezza)
atoms = graphene_nanoribbon(M, Ny,
                            type='zigzag',
                            saturated=True,
                            vacuum=10.0)  # vacuum nelle direzioni non periodiche

# Di solito il ribbon è nel piano x–y con vacuum in z,
# ma dipende dalla versione. Per sicurezza, rendiamo periodica
# solo la prima direzione (x).
atoms.pbc = (True, False, False)

# (opzionale) centra la ribbon rispetto all'origine nella direzione trasversale
pos = atoms.get_positions()
# NON sappiamo ancora se la larghezza è lungo y o z: guardiamo quale ha estensione maggiore
span_y = pos[:, 1].max() - pos[:, 1].min()
span_z = pos[:, 2].max() - pos[:, 2].min()

if span_y > span_z:
    # larghezza lungo y
    pos[:, 1] -= pos[:, 1].mean()
else:
    # larghezza lungo z
    pos[:, 2] -= pos[:, 2].mean()

atoms.set_positions(pos)

# --- DEBUG geometria ---
pos = atoms.get_positions()
print("=== NUOVA GEOMETRIA ZGNR ===")
print("Number of atoms:", len(atoms))
print("Cell (Å):")
print(atoms.cell)
print("PBC:", atoms.pbc)
print("y min / max:", pos[:, 1].min(), pos[:, 1].max())
print("z min / max:", pos[:, 2].min(), pos[:, 2].max())
print("First 10 positions (x, y, z):")
print(pos[:10])
print("=============================\n")

# Salva con un nome chiaro
fname = f'zgnr_Ny{Ny}_M{M}_ribbon.traj'
write(fname, atoms)
print(f"Written ZGNR structure to '{fname}'")
