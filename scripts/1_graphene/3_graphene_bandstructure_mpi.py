from gpaw import GPAW, PW
from ase.dft.kpoints import bandpath
from ase.parallel import parprint
from ase.build import graphene
import numpy as np


# -----------------------------------------------------
# 1. Load SCF ground-state
# -----------------------------------------------------
#calc_scf = GPAW('graphene_scf.gpw')
#calc_relaxed = GPAW('graphene_relaxed.gpw')
atoms = graphene()
atoms.center(vacuum=8.0, axis=2)
calc_scf = GPAW(
    mode=PW(600),               # same plane-wave basis as relaxation
    xc='PBE',
    kpts=(18, 18, 1),           # denser grid for accurate density
    txt='graphene_scf.txt',
    occupations={'name': 'fermi-dirac', 'width': 0.01},
)
#atoms = calc_scf.atoms

print("Loaded SCF: graphene_scf.gpw")


# -----------------------------------------------------
# 2. Build band path (Γ–K–M–Γ)
# -----------------------------------------------------
path = bandpath(
    [('G', 'K'), ('K', 'M'), ('M', 'G')],
    cell=atoms.cell,
    npoints=200
)

kpts = path.kpts
x, xticks, _ = path.get_linear_kpoint_axis()
print(f"Generated {len(kpts)} k-points for band path")


# -----------------------------------------------------
# 3. Create non-SCF calculator with *native GPAW parallelization*
# -----------------------------------------------------
calc = calc_scf.fixed_density(
    kpts=kpts,
    symmetry='off',
    parallel={'kpt': True},     # <- OFFICIAL PARALLELIZATION
    txt='graphene_bands.txt'
)

# GPAW distributes k-points over MPI ranks automatically
calc.get_potential_energy()

#
# # -----------------------------------------------------
# # 4. Collect eigenvalues (automatically distributed)
# # -----------------------------------------------------
# N = len(kpts)
# nbands = calc.get_number_of_bands()
#
# # Each rank reads only its assigned k-points
# local_eigs = []
# local_indices = []
#
# for ik in range(N):
#     if ik % calc.wfs.world.size == calc.wfs.world.rank:
#         ev = calc.get_eigenvalues(kpt=ik)
#         local_eigs.append(ev)
#         local_indices.append(ik)
#
# # Gather all on master rank
# all_eigs = calc.wfs.world.gather(local_eigs, 0)
# all_indices = calc.wfs.world.gather(local_indices, 0)
#
#
# # -----------------------------------------------------
# # 5. Master assembles in correct order and saves file
# # -----------------------------------------------------
# if calc.wfs.world.rank == 0:
#     eigs = np.zeros((N, nbands))
#
#     # Put pieces into correct positions
#     for rank_list, rank_inds in zip(all_eigs, all_indices):
#         for ev, ik in zip(rank_list, rank_inds):
#             eigs[ik, :] = ev
#
#     fermi = calc.get_fermi_level()
#
#     np.savez(
#         'graphene_bands_raw.npz',
#         eigs=eigs,
#         fermi=fermi,
#         x=x,
#         xticks=xticks
#     )
#
#     parprint("Saved: graphene_bands_raw.npz")
#     parprint("Band structure calculation complete.")
