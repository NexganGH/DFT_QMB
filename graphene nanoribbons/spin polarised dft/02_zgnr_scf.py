# 02_zgnr_scf.py
#
# Spin-polarized DFT for a zigzag graphene nanoribbon (ZGNR)
# Geometry must be generated beforehand by 01_zgnr_geometry.py

from ase.io import read
from gpaw import GPAW, PW, FermiDirac
import numpy as np
import config_zgnr as cfg


def set_initial_edge_magmoms(atoms, m0, edge_tol):
    """
    Assign ±m0 to C atoms at the physical edges along x.

    IMPORTANT:
    We detect edges using ONLY carbon atoms (ignore vacuum),
    otherwise the min/max over all positions would include the
    vacuum padding, and no atom would be near that minimum.
    """
    pos = atoms.get_positions()
    x = pos[:, 0]
    symbols = atoms.get_chemical_symbols()

    # Only carbon atoms matter for edge detection
    C_mask = np.array([s == 'C' for s in symbols])
    xC = x[C_mask]

    xmin_edge = xC.min()   # left physical edge (C)
    xmax_edge = xC.max()   # right physical edge (C)

    print(f"Detected C-edge positions: xmin_edge = {xmin_edge:.3f} Å, "
          f"xmax_edge = {xmax_edge:.3f} Å")

    magmoms = np.zeros(len(atoms), dtype=float)

    for i, (sym, xi) in enumerate(zip(symbols, x)):
        if sym != 'C':
            # H (or others) -> non-magnetic
            magmoms[i] = 0.0
            continue

        if abs(xi - xmin_edge) < edge_tol:
            magmoms[i] = +m0
        elif abs(xi - xmax_edge) < edge_tol:
            magmoms[i] = -m0
        else:
            magmoms[i] = 0.0

    atoms.set_initial_magnetic_moments(magmoms)
    return magmoms


def make_calculator(atoms):
    """
    Build GPAW calculator (PW or LCAO) according to config_zgnr.
    """
    kpts = (1, 1, cfg.kpts_z)

    if cfg.use_pw:
        calc = GPAW(
            mode=PW(cfg.pw_ecut),
            xc=cfg.xc,
            kpts=kpts,
            occupations=FermiDirac(cfg.fermi_width),
            spinpol=True,
            symmetry='off',
            txt='zgnr_scf_pw.txt'
        )
    else:
        calc = GPAW(
            mode='lcao',
            basis=cfg.lcao_basis,
            xc=cfg.xc,
            kpts=kpts,
            occupations=FermiDirac(cfg.fermi_width),
            spinpol=True,
            symmetry='off',
            txt='zgnr_scf_lcao.txt'
        )

    # recommended modern syntax
    atoms.calc = calc
    return calc


def main():
    # 1. Load geometry from step 01
    atoms = read(cfg.geom_traj)
    pos = atoms.get_positions()

    print("=== GEOMETRIA PRIMA DI SCF ===")
    print("Cell:", atoms.cell)
    print("PBC:", atoms.pbc)
    print("x span:", pos[:, 0].min(), "→", pos[:, 0].max())
    print("y span:", pos[:, 1].min(), "→", pos[:, 1].max())
    print("z span:", pos[:, 2].min(), "→", pos[:, 2].max())
    print("================================\n")

    # 2. Initial AF edge moments
    mag_init = set_initial_edge_magmoms(
        atoms,
        m0=cfg.spin_seed,
        edge_tol=cfg.edge_tol
    )
    print("Initial magmoms (first 20):", mag_init[:20])
    print("Initial total magmom:", mag_init.sum(), "\n")

    # 3. Attach GPAW calculator
    calc = make_calculator(atoms)

    # 4. Geometry relaxation OR just SCF
    if cfg.do_relax:
        from ase.optimize import BFGS
        print("Starting geometry relaxation (BFGS)...")
        dyn = BFGS(atoms, logfile='zgnr_relax.log')
        dyn.run(fmax=cfg.fmax, steps=cfg.max_steps)
        print("Relaxation finished.\n")
    else:
        print("No relaxation: running single SCF on fixed geometry...")
        atoms.get_potential_energy()
        print("SCF finished.\n")

    # 5. Final total energy
    E_tot = atoms.get_potential_energy()
    print("Final total energy (eV):", E_tot)

    # 6. Final magnetic moments and positions
    mag_final = atoms.get_magnetic_moments()
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    print("Final magmoms (first 20):", mag_final[:20])
    print("Final total magmom:", mag_final.sum(), "\n")

    # 7. Save SCF state (.gpw)
    calc.write(cfg.gpw_file, mode='all')
    print(f"SCF state written to '{cfg.gpw_file}'")

    # 8. Save atomic data for post-processing (.npz)
    np.savez(
        cfg.mag_atoms_npz,
        positions=pos,
        symbols=np.array(symbols),
        magmoms=mag_final,
        cell=atoms.cell.array,
        pbc=np.array(atoms.pbc, dtype=bool),
        E_tot=E_tot
    )
    print(f"Atomic positions & magnetizations saved to '{cfg.mag_atoms_npz}'")


if __name__ == "__main__":
    main()
