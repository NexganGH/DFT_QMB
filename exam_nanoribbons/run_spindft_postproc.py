# run_spindft_postproc.py
import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_spindft import MODE, NY_SINGLE, NY_LIST, BASE_OUTDIR
from spindft_paths import case_dir, ensure_dirs

from spindft_pretty_plots_core import (
    pretty_bands_from_npz,
    pretty_magnetization_profile_from_tbfit,
    lattice_map_from_tbfit,
)


def case_list():
    if MODE == "single":
        return [NY_SINGLE]
    return list(NY_LIST)


def _find_one(patterns):
    """Return first existing file matching any glob pattern, else None."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main():
    for Ny in case_list():
        case = case_dir(BASE_OUTDIR, MODE, Ny)
        dirs = ensure_dirs(case)

        # ✅ robust: find the actual files inside the case folders
        bands_npz = _find_one([
            os.path.join(dirs["data"], "*_bands_pi.npz"),
        ])

        tbfit_npz = _find_one([
            os.path.join(dirs["tbfit"], "*_tbfit.npz"),
            os.path.join(dirs["tbfit"], "*_tb_fit.npz"),
        ])

        # Output folder
        pretty_dir = os.path.join(dirs["plots"], "pretty")
        os.makedirs(pretty_dir, exist_ok=True)

        # 1) Pretty bands (colored + gray)
        if bands_npz and os.path.isfile(bands_npz):
            out_col = os.path.join(pretty_dir, "bands_colored.png")
            out_gra = os.path.join(pretty_dir, "bands_gray.png")
            pretty_bands_from_npz(bands_npz, out_col, out_gra, Ny_label=Ny)
            print("[OK] bands ->", out_col, out_gra)
        else:
            print("[SKIP] missing bands npz in:", dirs["data"])

        # 2) Pretty magnetization TB vs DFT + lattice maps
        if tbfit_npz and os.path.isfile(tbfit_npz):
            out_m = os.path.join(pretty_dir, "magn_profile_pretty.png")
            pretty_magnetization_profile_from_tbfit(tbfit_npz, out_m, Ny_label=Ny)
            print("[OK] magn profile ->", out_m)

            out_tb = os.path.join(pretty_dir, "tb_mag_lattice.png")
            out_df = os.path.join(pretty_dir, "diff_mag_lattice_DFT_minus_TB.png")
            lattice_map_from_tbfit(tbfit_npz, out_tb, mode="tb")
            lattice_map_from_tbfit(tbfit_npz, out_df, mode="diff")
            print("[OK] lattice maps ->", out_tb, out_df)
        else:
            print("[SKIP] missing tbfit npz in:", dirs["tbfit"])


if __name__ == "__main__":
    main()
