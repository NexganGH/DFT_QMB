# bands.py
from gpaw import GPAW

calc_loaded = GPAW('gnr_pw.gpw')
atoms = calc_loaded.atoms

nel = calc_loaded.get_number_of_electrons()
nbands = 66#int(nel/2) + 20

print("Natoms:", len(atoms))
print("Electrons:", calc_loaded.get_number_of_electrons())
print("PW cutoff:", getattr(calc_loaded.wfs, 'ecut', None))
print("Grid shape:", calc_loaded.wfs.gd.N_c)
print("Spin polarized:", calc_loaded.get_number_of_spins())


calc_fixed = calc_loaded.fixed_density(
    nbands=nbands,
    symmetry='off',
    kpts={'path': 'GX', 'npoints': 30},
    convergence={'bands': 4},
    txt='bands_fixed_density.txt'
)

print('Done!')
calc_fixed.write('bands_fixed.gpw')
