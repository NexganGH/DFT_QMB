#!/usr/bin/env python3
# autopilot_pw.py
#
# Sweep Ny = 1,2,4,6,8 for spin-polarised ZGNR DFT with plane waves.
#
# Folder layout expected:
#
#   /.../graphene nanoribbons/
#       spin polarised dft PlaneWaves/
#           config_zgnr.py
#           01_zgnr_geometry.py
#           02_zgnr_scf.py
#           03_zgnr_magnetization.py
#           04_zgnr_bands_pi.py
#
#       spin pol dft autopilot planewaves/
#           autopilot_pw.py   <-- this file
#
# Results will be collected in:
#   spin pol dft autopilot planewaves/zgnr_sweep_results/Ny1/
#   spin pol dft autopilot planewaves/zgnr_sweep_results/Ny2/
#   ...
# Each Ny-folder will contain: gpw, traj, xyz, npz, png, txt logs, etc.
#
# You can later add your smoothing / fitting scripts directly inside each NyX folder.

import os
import sys
import shutil
import importlib.util

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def load_module_from_path(path, name):
    """Dynamically load a module from an arbitrary .py path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_cfg_for_Ny(cfg, Ny):
    """
    Override key config_zgnr parameters for a given Ny.
    We recompute all filename patterns so that they match the ones
    in your original config, but with the new Ny.
    """
    cfg.Ny = Ny
    cfg.Ny_bins = Ny

    # Make sure we're using plane waves
    cfg.use_pw = True

    # Geometry & basic output names
    cfg.geom_traj = f"zgnr_Ny{Ny}_M{cfg.M}_zperiodic.traj"
    cfg.geom_xyz  = f"zgnr_Ny{Ny}_M{cfg.M}_zperiodic.xyz"
    cfg.gpw_file  = f"zgnr_Ny{Ny}_M{cfg.M}_scf.gpw"

    # Magnetisation files
    cfg.mag_atoms_npz   = f"zgnr_Ny{Ny}_M{cfg.M}_mag_atoms.npz"
    cfg.mag_profile_npz = f"zgnr_Ny{Ny}_M{cfg.M}_mag_profile.npz"
    cfg.mag_profile_png = f"zgnr_Ny{Ny}_M{cfg.M}_mag_profile.png"
    cfg.mag_AB_npz      = f"zgnr_Ny{Ny}_M{cfg.M}_mag_AB_strands.npz"
    cfg.mag_AB_png      = f"zgnr_Ny{Ny}_M{cfg.M}_mag_AB_strands.png"

    # Band-structure files
    cfg.bands_pi_npz   = f"zgnr_Ny{Ny}_M{cfg.M}_bands_pi.npz"
    cfg.bands_pi_png   = f"zgnr_Ny{Ny}_M{cfg.M}_bands_pi.png"
    cfg.bands_full_png = f"zgnr_Ny{Ny}_M{cfg.M}_bands_full.png"

    # TB fit files (not used in this autopilot, but keep consistent)
    cfg.tb_fit_npz       = f"zgnr_Ny{Ny}_M{cfg.M}_tb_fit.npz"
    cfg.tb_fit_bands_png = f"zgnr_Ny{Ny}_M{cfg.M}_tb_fit_bands.png"
    cfg.tb_fit_mag_png   = f"zgnr_Ny{Ny}_M{cfg.M}_tb_fit_mag.png"


def collect_outputs(base_dft_dir, Ny_dir, cfg):
    """
    Move all relevant files for this Ny from the DFT folder to the Ny-specific folder.
    Only moves files that actually exist.
    """
    # Primary outputs from config
    candidates = [
        cfg.geom_traj,
        cfg.geom_xyz,
        cfg.gpw_file,
        cfg.mag_atoms_npz,
        cfg.mag_profile_npz,
        cfg.mag_profile_png,
        cfg.mag_AB_npz,
        cfg.mag_AB_png,
        cfg.bands_pi_npz,
        cfg.bands_pi_png,
        cfg.bands_full_png,
        cfg.tb_fit_npz,
        cfg.tb_fit_bands_png,
        cfg.tb_fit_mag_png,
    ]

    # Logs / txt files with fixed names used in your scripts
    candidates += [
        "zgnr_scf_pw.txt",
        "zgnr_scf_lcao.txt",
        "zgnr_relax.log",
        "zgnr_bands.txt",
    ]

    for fname in candidates:
        src = os.path.join(base_dft_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(Ny_dir, fname)
            print(f"    → moving '{fname}' → '{Ny_dir}'")
            shutil.move(src, dst)


# ----------------------------------------------------------------------
#  MAIN AUTOPILOT
# ----------------------------------------------------------------------

def main():
    # ---- 0. Locate folders ----
    this_dir = os.path.abspath(os.path.dirname(__file__))

    # Folder with the DFT scripts and config_zgnr.py
    base_dft_dir = os.path.abspath(
        os.path.join(this_dir, "..", "spin polarised dft PlaneWaves")
    )

    if not os.path.isdir(base_dft_dir):
        raise RuntimeError(
            f"Could not find DFT folder at:\n  {base_dft_dir}\n"
            "Check that your directory names match."
        )

    # Add that folder to sys.path so we can import config_zgnr
    if base_dft_dir not in sys.path:
        sys.path.insert(0, base_dft_dir)

    try:
        import config_zgnr as cfg
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Could not import 'config_zgnr'. Make sure there is a file "
            "'config_zgnr.py' inside 'spin polarised dft PlaneWaves/'.\n"
            "If your file is named differently (e.g. config_zgnr_LCAO.py), "
            "rename or duplicate it to 'config_zgnr.py'."
        ) from e

    # Load the step modules
    geom_mod = load_module_from_path(
        os.path.join(base_dft_dir, "01_zgnr_geometry.py"), "zgnr_geometry"
    )
    scf_mod = load_module_from_path(
        os.path.join(base_dft_dir, "02_zgnr_scf.py"), "zgnr_scf"
    )
    mag_mod = load_module_from_path(
        os.path.join(base_dft_dir, "03_zgnr_magnetization.py"), "zgnr_magnetization"
    )
    bands_mod = load_module_from_path(
        os.path.join(base_dft_dir, "04_zgnr_bands_pi.py"), "zgnr_bands_pi"
    )

    # List of Ny to sweep
    Ny_list = [1, 2, 4, 6, 8]

    # Results root inside the autopilot folder
    results_root = os.path.join(this_dir, "zgnr_sweep_results")
    os.makedirs(results_root, exist_ok=True)

    print("====================================================")
    print(" ZGNR spin-polarised DFT sweep with plane waves")
    print(" DFT folder:", base_dft_dir)
    print(" Results   :", results_root)
    print(" Ny values :", Ny_list)
    print("====================================================\n")

    orig_cwd = os.getcwd()

    # Choose loop implementation depending on tqdm availability
    if HAS_TQDM:
        iterator = tqdm(Ny_list,
                        desc="ZGNR sweep",
                        unit="case",
                        dynamic_ncols=True)
    else:
        iterator = Ny_list

    for Ny in iterator:
        if HAS_TQDM:
            iterator.set_postfix({"Ny": Ny, "step": "init"}, refresh=True)

        print(f"\n\n========== Starting Ny = {Ny} ==========")

        # Create folder for this Ny
        Ny_dir = os.path.join(results_root, f"Ny{Ny}")
        os.makedirs(Ny_dir, exist_ok=True)

        # Update configuration for this Ny
        update_cfg_for_Ny(cfg, Ny)

        print("Using configuration:")
        print(f"  Ny         = {cfg.Ny}")
        print(f"  M          = {cfg.M}")
        print(f"  use_pw     = {cfg.use_pw}")
        print(f"  pw_ecut    = {cfg.pw_ecut} eV")
        print(f"  kpts_z     = {cfg.kpts_z}")
        print(f"  npoints_kpath = {cfg.npoints_kpath}")
        print(f"  do_relax   = {cfg.do_relax}")
        print(f"  geom_traj  = {cfg.geom_traj}")
        print(f"  gpw_file   = {cfg.gpw_file}")
        print("----------------------------------------")

        # Work inside the DFT folder so all relative paths match
        os.chdir(base_dft_dir)

        # ---- Step 1: geometry ----
        print(f"[Ny={Ny}] Step 1/4: geometry / nanoribbon construction")
        if HAS_TQDM:
            iterator.set_postfix({"Ny": Ny, "step": "geometry"}, refresh=True)
        geom_mod.main()

        # ---- Step 2: SCF (+ optional relaxation) ----
        if cfg.do_relax:
            print(f"[Ny={Ny}] Step 2/4: SCF with geometry relaxation (BFGS)")
        else:
            print(f"[Ny={Ny}] Step 2/4: SCF on fixed geometry (no relax)")
        if HAS_TQDM:
            iterator.set_postfix({"Ny": Ny, "step": "SCF/relax"}, refresh=True)
        scf_mod.main()

        # ---- Step 3: magnetisation post-processing ----
        print(f"[Ny={Ny}] Step 3/4: magnetisation profiles (bins + A/B strands)")
        if HAS_TQDM:
            iterator.set_postfix({"Ny": Ny, "step": "magnetisation"}, refresh=True)
        mag_mod.main()

        # ---- Step 4: band structure ----
        print(f"[Ny={Ny}] Step 4/4: band structure along z (π window)")
        if HAS_TQDM:
            iterator.set_postfix({"Ny": Ny, "step": "bands"}, refresh=True)
        bands_mod.main()

        # ---- Collect & move outputs ----
        print(f"[Ny={Ny}] Collecting output files into '{Ny_dir}' ...")
        collect_outputs(base_dft_dir, Ny_dir, cfg)

        # Go back to original cwd (autopilot folder)
        os.chdir(orig_cwd)

        print(f"[Ny={Ny}] DONE.")
        print("============================================")

    print("\nAll Ny cases completed.")
    print(f"Results organised under: {results_root}")


if __name__ == "__main__":
    main()
