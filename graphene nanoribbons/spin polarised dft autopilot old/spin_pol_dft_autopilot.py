# autopilot_zgnr.py
#
# Folder structure assumed:
#
#   <ROOT>/
#     mft/
#       magnetisation_mft.py  (and other TB code)
#
#     spin polarised dft LCAO/
#       config_zgnr_LCAO.py
#       01_zgnr_geometry.py
#       02_zgnr_scf.py
#       03_zgnr_magnetization.py
#       04_zgnr_bands_pi.py
#       05_fit.py
#
#     spin polarised dft LCAO autopilot/
#       autopilot_zgnr.py  <-- THIS FILE
#
# This script:
#   - sweeps Ny_list = [1, 2, 4, 6, 8]
#   - for each Ny, creates:
#       spin polarised dft LCAO autopilot/zgnr_sweep_results/Ny<Ny>/
#     and redirects ALL cfg.* filenames into that folder.
#   - runs the 5 steps in order.
#   - tracks progress with a tqdm bar + detailed messages.
#   - writes a global fit_summary.txt with t_opt, U_opt for each Ny.

import os
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm
import importlib.util


def _load_module(module_name: str, filename: str):
    """Load a Python file as a module from an arbitrary path."""
    spec = importlib.util.spec_from_file_location(module_name, filename)
    mod = importlib.util.module_from_spec(spec)
    # Register module name in sys.modules (good practice)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------
# 1. Figure out relevant directories from this file location
# ---------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))      # .../spin polarised dft LCAO autopilot
ROOT = os.path.dirname(HERE)                           # parent containing the three folders

DFT_DIR = os.path.join(ROOT, "spin polarised dft LCAO")     # where config_zgnr + 01..05 are
MFT_DIR = os.path.join(ROOT, "mft")                    # TB code (used inside 05_fit.py)

# Make sure Python can import from these places
for p in [DFT_DIR, ROOT, MFT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Now we can import config_zgnr in the usual way
import config_zgnr as cfg

# Load the step scripts from the DFT_DIR
step01 = _load_module("zgnr_geom", os.path.join(DFT_DIR, "01_zgnr_geometry.py"))
step02 = _load_module("zgnr_scf", os.path.join(DFT_DIR, "02_zgnr_scf.py"))
step03 = _load_module("zgnr_mag", os.path.join(DFT_DIR, "03_zgnr_magnetization.py"))
step04 = _load_module("zgnr_bands", os.path.join(DFT_DIR, "04_zgnr_bands_pi.py"))
step05 = _load_module("zgnr_fit", os.path.join(DFT_DIR, "05_fit.py"))


# ---------------------------------------------------------
# 2. Rewrite config_zgnr for each Ny and output directory
# ---------------------------------------------------------

def configure_config_for_case(Ny: int, case_dir: str):
    """
    Update config_zgnr (cfg) in place for a given Ny and output directory.

    - Sets cfg.Ny and cfg.Ny_bins.
    - Redirects ALL filenames in cfg to live inside case_dir.
    """
    os.makedirs(case_dir, exist_ok=True)

    cfg.Ny = Ny
    cfg.Ny_bins = Ny  # keep your original choice: Ny_bins ~ number of strands

    Ny_str = f"Ny{cfg.Ny}_M{cfg.M}"

    # Use absolute paths to avoid any CWD issues
    case_dir_abs = os.path.abspath(case_dir)

    # --- Geometry / SCF output ---
    cfg.geom_traj = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_zperiodic.traj")
    cfg.geom_xyz  = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_zperiodic.xyz")
    cfg.gpw_file  = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_scf.gpw")

    # --- Magnetization ---
    cfg.mag_atoms_npz   = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_mag_atoms.npz")
    cfg.mag_profile_npz = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_mag_profile.npz")
    cfg.mag_profile_png = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_mag_profile.png")
    cfg.mag_AB_npz      = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_mag_AB_strands.npz")
    cfg.mag_AB_png      = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_mag_AB_strands.png")

    # --- Bands ---
    cfg.bands_pi_npz   = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_bands_pi.npz")
    cfg.bands_pi_png   = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_bands_pi.png")
    cfg.bands_full_png = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_bands_full.png")

    # --- TB fit ---
    cfg.tb_fit_npz       = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_tb_fit.npz")
    cfg.tb_fit_bands_png = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_tb_fit_bands.png")
    cfg.tb_fit_mag_png   = os.path.join(case_dir_abs, f"zgnr_{Ny_str}_tb_fit_mag.png")


# ---------------------------------------------------------
# 3. Main sweep logic
# ---------------------------------------------------------

def main():
    # Set of Ny values to run
    Ny_list = [1, 2, 4, 6, 8]

    # Root folder where all per-Ny folders will live
    sweep_root = os.path.join(HERE, "zgnr_sweep_results")
    os.makedirs(sweep_root, exist_ok=True)

    summary_lines = []

    print("Starting ZGNR spin-polarised DFT + TB fit sweep.")
    print("Ny values:", Ny_list)
    print("Global results folder:", sweep_root)

    for Ny in tqdm(Ny_list, desc="ZGNR sweep", unit="case"):
        case_dir = os.path.join(sweep_root, f"Ny{Ny}")
        tqdm.write("\n" + "=" * 70)
        tqdm.write(f"Ny = {Ny}  |  case directory: {case_dir}")
        tqdm.write("=" * 70)

        # Re-configure paths for this Ny
        configure_config_for_case(Ny, case_dir)

        t_case_start = time.time()

        try:
            # ---- Step 1: Geometry ----
            tqdm.write(f"[Ny={Ny}] Step 1/5: Geometry (01_zgnr_geometry.py)")
            step01.main()

            # ---- Step 2: SCF + relaxation ----
            tqdm.write(f"[Ny={Ny}] Step 2/5: SCF + relaxation (02_zgnr_scf.py)")
            tqdm.write(f"[Ny={Ny}]   → Running spin-polarised DFT with xc={cfg.xc}, kpts_z={cfg.kpts_z}")
            step02.main()

            # ---- Step 3: Magnetization analysis ----
            tqdm.write(f"[Ny={Ny}] Step 3/5: Magnetization (03_zgnr_magnetization.py)")
            step03.main()

            # ---- Step 4: π band structure ----
            tqdm.write(f"[Ny={Ny}] Step 4/5: π bands (04_zgnr_bands_pi.py)")
            step04.main()

            # ---- Step 5: TB fit ----
            tqdm.write(f"[Ny={Ny}] Step 5/5: TB fit (05_fit.py)")
            tqdm.write(f"[Ny={Ny}]   → Fitting (t, U) to DFT central π bands...")
            step05.main()

            t_case_end = time.time()
            elapsed_min = (t_case_end - t_case_start) / 60.0
            tqdm.write(f"[Ny={Ny}] DONE. Elapsed time ≈ {elapsed_min:.1f} minutes.")

            # Try to load t_opt, U_opt for this Ny and store in summary
            try:
                data = np.load(cfg.tb_fit_npz, allow_pickle=True)
                t_opt = float(data["t_opt"])
                U_opt = float(data["U_opt"])
                line = f"Ny = {Ny:2d} : t_opt = {t_opt:.4f} eV,  U_opt = {U_opt:.4f} eV\n"
                summary_lines.append(line)
                tqdm.write("[Ny={}]   Fit parameters: {}".format(Ny, line.strip()))
            except Exception as e:
                msg = f"Ny = {Ny:2d} : ERROR reading fit parameters: {e}\n"
                summary_lines.append(msg)
                tqdm.write(f"[Ny={Ny}] WARNING: {msg.strip()}")

        except Exception as e:
            tqdm.write(f"[Ny={Ny}] ERROR during pipeline: {e}")
            traceback.print_exc()
            summary_lines.append(f"Ny = {Ny:2d} : PIPELINE FAILED: {e}\n")
            # continue to next Ny

    # Write global summary file
    summary_path = os.path.join(sweep_root, "fit_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)

    print("\n=== SWEEP FINISHED ===")
    print("Global fit summary:", summary_path)
    print("All per-Ny results in:", sweep_root)


if __name__ == "__main__":
    main()
