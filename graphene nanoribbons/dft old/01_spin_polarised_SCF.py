from gpaw import GPAW, restart, FermiDirac

# ---------- MAGNETISATION HELPERS (C ONLY) ----------

def set_af_on_carbons(atoms, m=1.0):
    """Antiferromagnetic pattern on carbon atoms only: +m, -m, +m, -m, ..."""
    symbols = atoms.get_chemical_symbols()
    magmoms = []
    spin = +m
    for s in symbols:
        if s == 'C':
            magmoms.append(spin)
            spin *= -1.0
        else:
            magmoms.append(0.0)
    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


def set_fm_on_carbons(atoms, m=1.0):
    """Ferromagnetic pattern on carbon atoms only: all +m."""
    symbols = atoms.get_chemical_symbols()
    magmoms = [(m if s == 'C' else 0.0) for s in symbols]
    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


def set_pm_on_carbons(atoms):
    """Paramagnetic start: all atoms start with 0.0."""
    magmoms = [0.0] * len(atoms)
    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


# ----------------- MAIN SCRIPT -----------------

# 1. Restart from relaxed structure
atoms, calc0 = restart('chain_relaxed.gpw')

# 2. Enforce full PBC (vacuum should already be in the cell from the builder)
atoms.pbc = (True, True, True)
atoms.center()  # optional: just center atoms in the cell without changing sizes

print("Cell vectors (Å):\n", atoms.cell)
print("PBC:", atoms.pbc)

# 3. Choose initial magnetisation pattern on C atoms only
mode = 'FM'   # change to 'AF', 'FM', or 'PM'

if mode == 'AF':
    magmoms_init = set_af_on_carbons(atoms, m=1.0)
elif mode == 'FM':
    magmoms_init = set_fm_on_carbons(atoms, m=1.0)
elif mode == 'PM':
    magmoms_init = set_pm_on_carbons(atoms)
else:
    raise ValueError("Unknown mode. Use 'AF', 'FM', or 'PM'.")

symbols = atoms.get_chemical_symbols()

print(f"\n====== INITIAL MAGNETISATION (mode={mode}) ======")
for i, (s, m) in enumerate(zip(symbols, magmoms_init)):
    print(f"Atom {i:2d} ({s}): m_init = {m: .3f} μ_B")

# 4. Spin-polarised SCF with full PBC
calc_spin_pbc = GPAW(
    mode='fd',
    h=0.20,
    xc='PBE',
    kpts=(1, 1, 60),  # dense along z; x,y have vacuum but still "periodic"
    occupations=FermiDirac(0.01),
    spinpol=True,
    txt=f'chain_spin_{mode}_pbc.txt'
)

atoms.calc = calc_spin_pbc
energy = atoms.get_potential_energy()
print(f'\nTotal energy (spin, full PBC, start={mode}):', energy, 'eV')

# 5. Final magnetisation
total_mag = atoms.get_magnetic_moment()
site_mags = atoms.get_magnetic_moments()

print("\n====== FINAL (CONVERGED) MAGNETISATION ======")
print("Total magnetic moment:", total_mag, "μ_B")
for i, (s, m) in enumerate(zip(symbols, site_mags)):
    print(f"Atom {i:2d} ({s}): m_final = {m: .4f} μ_B")
print("=====================================\n")

# 6. Save converged calculator for bands/DOS
calc_spin_pbc.write(f'chain_spin_{mode}_pbc.gpw')
