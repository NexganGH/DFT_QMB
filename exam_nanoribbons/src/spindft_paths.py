# spindft_paths.py
import os

def case_dir(base_outdir: str, mode: str, Ny: int) -> str:
    return os.path.join(base_outdir, mode, f"Ny{Ny}")

def ensure_dirs(case: str) -> dict:
    d = {
        "case": case,
        "dft": os.path.join(case, "dft"),
        "data": os.path.join(case, "data"),
        "plots": os.path.join(case, "plots"),
        "logs": os.path.join(case, "logs"),
        "tbfit": os.path.join(case, "tbfit"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d

def relabel_cfg_files(dirs: dict, Ny: int, M: int) -> dict:
    # Standardized filenames (always inside dirs)
    files = {
        "geom_traj": os.path.join(dirs["dft"], f"zgnr_Ny{Ny}_M{M}.traj"),
        "geom_xyz":  os.path.join(dirs["dft"], f"zgnr_Ny{Ny}_M{M}.xyz"),
        "gpw":       os.path.join(dirs["dft"], f"zgnr_Ny{Ny}_M{M}_scf.gpw"),

        "mag_atoms_npz":   os.path.join(dirs["data"], f"zgnr_Ny{Ny}_M{M}_mag_atoms.npz"),
        "mag_profile_npz": os.path.join(dirs["data"], f"zgnr_Ny{Ny}_M{M}_mag_profile.npz"),
        "mag_AB_npz":      os.path.join(dirs["data"], f"zgnr_Ny{Ny}_M{M}_mag_AB_strands.npz"),

        "mag_profile_png": os.path.join(dirs["plots"], f"zgnr_Ny{Ny}_M{M}_mag_profile.png"),
        "mag_AB_png":      os.path.join(dirs["plots"], f"zgnr_Ny{Ny}_M{M}_mag_AB_strands.png"),

        "bands_full_png":  os.path.join(dirs["plots"], f"zgnr_Ny{Ny}_M{M}_bands_full.png"),
        "bands_pi_png":    os.path.join(dirs["plots"], f"zgnr_Ny{Ny}_M{M}_bands_pi.png"),
        "bands_pi_npz":    os.path.join(dirs["data"],  f"zgnr_Ny{Ny}_M{M}_bands_pi.npz"),

        "tbfit_npz":       os.path.join(dirs["tbfit"], f"zgnr_Ny{Ny}_M{M}_tb_fit.npz"),
        "tbfit_bands_png": os.path.join(dirs["tbfit"], f"zgnr_Ny{Ny}_M{M}_tb_fit_bands.png"),
        "tbfit_mag_png":   os.path.join(dirs["tbfit"], f"zgnr_Ny{Ny}_M{M}_tb_fit_mag.png"),
    }
    return files
