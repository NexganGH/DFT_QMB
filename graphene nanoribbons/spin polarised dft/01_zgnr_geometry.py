# 01_zgnr_geometry.py

from ase.build import graphene_nanoribbon
from ase.io import write
import numpy as np
import config_zgnr as cfg


def construct_zgnr():
    """
    ZGNR with:
      - width Ny (zigzag chains)  -> along x
      - length M (cells)          -> along z (periodic)

    ASE docs: graphene_nanoribbon creates a ribbon in the x–z plane,
    with the ribbon running along z. So we just:
      - build it,
      - add vacuum,
      - set PBC only along z.
    No extra rotations.
    """

    atoms = graphene_nanoribbon(
        cfg.Ny,            # width (zigzag chains)
        cfg.M,             # length (number of cells along z)
        type='zigzag',
        saturated=True,
        C_C=cfg.C_C,
        vacuum=cfg.vacuum_x   # vacuum added to non-periodic directions
    )

    # Make sure only z is periodic
    atoms.set_pbc((False, False, True))

    return atoms


def main():
    atoms = construct_zgnr()
    pos = atoms.get_positions()
    cell = atoms.cell.array

    span_x = pos[:, 0].max() - pos[:, 0].min()
    span_y = pos[:, 1].max() - pos[:, 1].min()
    span_z = pos[:, 2].max() - pos[:, 2].min()

    print("=== GEOMETRIA ZGNR ORIENTATA (ASE DEFAULT) ===")
    print("Ny =", cfg.Ny, "M =", cfg.M)
    print("Numero atomi:", len(atoms))
    print("Cell lengths (Å):", atoms.cell.lengths())
    print("Cell vectors (rows = a1,a2,a3):\n", cell)
    print("PBC:", atoms.pbc)
    print(f"span x: {span_x:.3f} Å")
    print(f"span y: {span_y:.3f} Å")
    print(f"span z: {span_z:.3f} Å")
    print(" (largest span should be z = periodic direction)")
    print("==============================================\n")

    write(cfg.geom_traj, atoms)
    write(cfg.geom_xyz, atoms)
    print(f"Scritta struttura in '{cfg.geom_traj}' e '{cfg.geom_xyz}'")


if __name__ == "__main__":
    main()
