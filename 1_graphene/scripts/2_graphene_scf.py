from gpaw import GPAW, PW

# --------------------------------------------------------
# Load relaxed structure (geometry + initial density)
# --------------------------------------------------------
calc_relaxed = GPAW('../outputs/graphene_relaxed.gpw')

atoms = calc_relaxed.atoms

# --------------------------------------------------------
# High-quality SCF calculation
# --------------------------------------------------------
calc_scf = GPAW(
    mode=PW(600),               # same plane-wave basis as relaxation
    xc='PBE',
    kpts=(18, 18, 1),           # denser grid for accurate density
    txt='graphene_scf.txt',
    occupations={'name': 'fermi-dirac', 'width': 0.01},
)

atoms.calc = calc_scf

print("Running SCF on relaxed graphene...")
energy = atoms.get_potential_energy()
fermi = calc_scf.get_fermi_level()

print(f"\nSCF total energy: {energy:.6f} eV")
print(f"Fermi level: {fermi:.6f} eV")

# --------------------------------------------------------
# Save for bands and DOS
# --------------------------------------------------------
calc_scf.write('graphene_scf.gpw')
print("Saved SCF wavefunctions to graphene_scf.gpw")
