from gpaw import GPAW, restart, FermiDirac


def set_af_on_carbons(atoms, m=1.0):
    """
    Set an antiferromagnetic pattern on carbon atoms only:
    C: +m, -m, +m, -m, ...
    H (and others): 0.0
    """
    symbols = atoms.get_chemical_symbols()
    magmoms = []

    spin = +m
    for s in symbols:
        if s == 'C':
            magmoms.append(spin)
            spin *= -1.0   # flip sign for next C
        else:
            magmoms.append(0.0)

    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


def set_fm_on_carbons(atoms, m=1.0):
    """
    Set a ferromagnetic pattern on carbon atoms only:
    C: +m
    H (and others): 0.0
    """
    symbols = atoms.get_chemical_symbols()
    magmoms = []

    for s in symbols:
        if s == 'C':
            magmoms.append(m)
        else:
            magmoms.append(0.0)

    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


def set_pm_on_carbons(atoms):
    """
    Paramagnetic start: all atoms (including C) start with 0.0
    """
    n = len(atoms)
    magmoms = [0.0] * n
    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


# ----------------- MAIN SCRIPT -----------------

# 1. Read relaxed structure
atoms, calc0 = restart('chain_relaxed.gpw')

# 2. Choose which initial configuration you want:
mode = 'FM'   # change to 'FM' or 'PM' as you like

if mode == 'AF':
    magmoms = set_af_on_carbons(atoms, m=1.0)
elif mode == 'FM':
    magmoms = set_fm_on_carbons(atoms, m=1.0)
elif mode == 'PM':
    magmoms = set_pm_on_carbons(atoms)
else:
    raise ValueError("Unknown mode. Use 'AF', 'FM', or 'PM'.")

# Print what you’ve set
symbols = atoms.get_chemical_symbols()
print(f"Initial magnetic moments (mode={mode}):")
for i, (s, m) in enumerate(zip(symbols, magmoms)):
    print(f"Atom {i:2d} ({s}): m_init = {m: .2f} μ_B")

# 3. Spin-polarised SCF calculation
calc_spin = GPAW(
    mode='fd',
    h=0.20,
    xc='PBE',
    kpts=(1, 1, 60),
    occupations=FermiDirac(0.01),
    spinpol=True,
    txt=f'chain_spin_{mode}.txt'
)

atoms.calc = calc_spin
energy = atoms.get_potential_energy()
print(f'Total energy (spin-polarised, start={mode}):', energy, 'eV')

# 4. Magnetisation analysis
total_mag = atoms.get_magnetic_moment()
site_mags = atoms.get_magnetic_moments()

print('Total magnetic moment:', total_mag, 'μ_B')
for i, (s, m) in enumerate(zip(symbols, site_mags)):
    print(f'Atom {i:2d} ({s}): mag = {m: .4f} μ_B')

# 5. Save converged calculator
calc_spin.write(f'chain_spin_{mode}.gpw')
