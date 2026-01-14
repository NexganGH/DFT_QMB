# spindft_geometry_core.py
from ase.build import graphene_nanoribbon
from ase.io import write

def construct_zgnr_atoms(Ny: int, M: int, C_C: float, vacuum: float, saturated: bool):
    atoms = graphene_nanoribbon(
        Ny,
        M,
        type="zigzag",
        saturated=saturated,
        C_C=C_C,
        vacuum=vacuum,
    )
    # only z periodic (ASE builds ribbon in x–z plane, periodic along z)
    atoms.set_pbc((False, False, True))
    return atoms

def write_geometry(atoms, traj_path: str, xyz_path: str):
    write(traj_path, atoms)
    write(xyz_path, atoms)
