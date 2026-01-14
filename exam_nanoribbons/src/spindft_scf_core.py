# spindft_scf_core.py
import numpy as np
from gpaw import GPAW, PW, FermiDirac

def set_initial_edge_magmoms(atoms, m0: float, edge_tol: float):
    pos = atoms.get_positions()
    x = pos[:, 0]
    symbols = atoms.get_chemical_symbols()

    C_mask = np.array([s == "C" for s in symbols])
    xC = x[C_mask]
    xmin_edge = xC.min()
    xmax_edge = xC.max()

    magmoms = np.zeros(len(atoms), dtype=float)
    for i, (sym, xi) in enumerate(zip(symbols, x)):
        if sym != "C":
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

def make_calculator(
    atoms,
    use_pw: bool,
    pw_ecut: float,
    lcao_basis: str,
    xc: str,
    kpts_z: int,
    fermi_width: float,
    txt_path: str,
):
    kpts = (1, 1, kpts_z)
    if use_pw:
        calc = GPAW(
            mode=PW(pw_ecut),
            xc=xc,
            kpts=kpts,
            occupations=FermiDirac(fermi_width),
            spinpol=True,
            symmetry="off",
            txt=txt_path,
        )
    else:
        calc = GPAW(
            mode="lcao",
            basis=lcao_basis,
            xc=xc,
            kpts=kpts,
            occupations=FermiDirac(fermi_width),
            spinpol=True,
            symmetry="off",
            txt=txt_path,
        )
    atoms.calc = calc
    return calc

def run_scf(atoms, do_relax: bool, fmax: float, max_steps: int):
    if do_relax:
        from ase.optimize import BFGS
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=fmax, steps=max_steps)
    else:
        atoms.get_potential_energy()
    return atoms.calc  # GPAW calculator attached
