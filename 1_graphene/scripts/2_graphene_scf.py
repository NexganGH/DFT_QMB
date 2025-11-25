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
    mode=PW(1000),               # same plane-wave basis as relaxation
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


# -----------------------------
# 3. Save important SCF data
# -----------------------------
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

output_path = '../outputs/graphene_scf_data.txt'
with open(output_path, 'w') as f:
    f.write("Graphene SCF Calculation Results\n")
    f.write("Timestamp: {}\n\n".format(timestamp))

    f.write("=== Energetics ===\n")
    f.write(f"SCF total energy: {energy:.6f} eV\n")
    f.write(f"Fermi level:      {fermi:.6f} eV\n")
    f.write("\n")

    f.write("=== Computational Parameters ===\n")
    f.write("Exchange–correlation: PBE\n")
    f.write("Basis: Plane waves\n")
    f.write("Cutoff: 600 eV\n")
    f.write("k-points: 18 × 18 × 1\n")
    f.write("Smearing: Fermi–Dirac, width = 0.01 eV\n")
    f.write("\n")

print(f"Saved SCF summary to: {output_path}\n")


# --------------------------------------------------------
# Save for bands and DOS
# --------------------------------------------------------
calc_scf.write('graphene_scf.gpw')
print("Saved SCF wavefunctions to graphene_scf.gpw")
