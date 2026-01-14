# run_spindft.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_spindft import (
    MODE, NY_SINGLE, NY_LIST,
    BASE_OUTDIR,
    M, C_C, VACUUM, SATURATED,
    USE_PW, PW_ECUT, LCAO_BASIS,
    XC, KPTS_Z, FERMI_WIDTH,
    SPIN_SEED, EDGE_TOL,
    DO_RELAX, FMAX, MAX_STEPS,
    NPOINTS_KPATH, NBANDS_BANDS,
    E_MIN_PI, E_MAX_PI
)

from spindft_paths import case_dir, ensure_dirs, relabel_cfg_files
from spindft_geometry_core import construct_zgnr_atoms, write_geometry
from spindft_scf_core import set_initial_edge_magmoms, make_calculator, run_scf
from spindft_magnetization_core import extract_magnetization
from spindft_bands_core import compute_bands_pi

def run_case(mode: str, Ny: int):
    case = case_dir(BASE_OUTDIR, mode, Ny)
    dirs = ensure_dirs(case)
    files = relabel_cfg_files(dirs, Ny=Ny, M=M)

    # 1) geometry
    atoms = construct_zgnr_atoms(Ny=Ny, M=M, C_C=C_C, vacuum=VACUUM, saturated=SATURATED)
    write_geometry(atoms, files["geom_traj"], files["geom_xyz"])

    # 2) initial edge moments + calculator
    set_initial_edge_magmoms(atoms, m0=SPIN_SEED, edge_tol=EDGE_TOL)

    txt_path = os.path.join(dirs["logs"], "zgnr_scf.txt")
    make_calculator(
        atoms,
        use_pw=USE_PW,
        pw_ecut=PW_ECUT,
        lcao_basis=LCAO_BASIS,
        xc=XC,
        kpts_z=KPTS_Z,
        fermi_width=FERMI_WIDTH,
        txt_path=txt_path,
    )

    # 3) scf (optional relax)
    calc = run_scf(atoms, do_relax=DO_RELAX, fmax=FMAX, max_steps=MAX_STEPS)
    calc.write(files["gpw"], mode="all")

    # 4) magnetization extraction (needs row_tol; pick something stable)
    # row_tol is geometry-dependent; 0.25 Å is a reasonable start for ASE ribbons
    extract_magnetization(
        gpw_path=files["gpw"],
        Ny_bins=Ny,
        row_tol=0.25,
        mag_profile_png=None,  # disable
        mag_AB_png=None,  # disable
        mag_profile_npz=files["mag_profile_npz"],
        mag_AB_npz=files["mag_AB_npz"],
    )

    # 5) bands
    compute_bands_pi(
        gpw_path=files["gpw"],
        npoints_kpath=NPOINTS_KPATH,
        nbands=NBANDS_BANDS,
        e_min_pi=E_MIN_PI,
        e_max_pi=E_MAX_PI,
        bands_full_png=None,  # disable
        bands_pi_png=None,  # disable
        bands_pi_npz=files["bands_pi_npz"],
    )

    print(f"[OK] Finished Ny={Ny} -> {case}")

def main():
    if MODE == "single":
        run_case("single", NY_SINGLE)
    elif MODE == "sweep_N":
        for Ny in NY_LIST:
            run_case("sweep_N", Ny)
    else:
        raise ValueError("MODE must be 'single' or 'sweep_N'")

if __name__ == "__main__":
    main()
