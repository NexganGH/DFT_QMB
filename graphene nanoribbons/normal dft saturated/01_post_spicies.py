from collections import Counter
from gpaw import GPAW
from config_zgnr import LABEL_BASE

gpw_file = f"{LABEL_BASE}_relaxed.gpw"
calc = GPAW(gpw_file)
atoms = calc.get_atoms()

symbols = atoms.get_chemical_symbols()
counts = Counter(symbols)

print("Atomic species counts:")
for s, n in counts.items():
    print(f"{s}: {n}")

print("Total atoms:", len(atoms))
