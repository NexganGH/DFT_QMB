from ase.build import graphene
from gpaw import GPAW

# -----------------------------
# Build graphene unit cell
# -----------------------------
atoms = graphene()              # 2 atoms, hexagonal cell
atoms.center(vacuum=8.0, axis=2)  # add vacuum along z so layers don't interact

print("Graphene structure:")
print(atoms)

# -----------------------------
# LDA SCF calculation
# -----------------------------
calc = GPAW(
    h=0.22,              # real-space grid spacing (in Å); smaller = more accurate, slower
    xc='PBE',            # <-- use LDA here
    kpts=(8, 8, 1),      # Brillouin zone sampling
    txt='graphene_lda_scf.txt'  # output log
)

atoms.center()
atoms.calc = calc

print("\nRunning LDA SCF on graphene...")
energy = atoms.get_potential_energy()
fermi = calc.get_fermi_level()

print(f"\nLDA SCF total energy: {energy:.6f} eV")
print(f"Fermi level (approx. Dirac point): {fermi:.6f} eV")

# Save the result to reuse later (bands, DOS, etc.)
calc.write('graphene_lda_scf.gpw')
print("\nSaved wavefunctions to graphene_lda_scf.gpw")
