# run_mf_lattice.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")
if INPUT_DIR not in sys.path:
    sys.path.insert(0, INPUT_DIR)

from inputs_mf_lattice import (
    BASE_DIR,
    N_MIN, N_MAX, LATTICE_MAX_N,
    FLIP_WIDTH_ORDER,
    N_REPEAT, DX_STEP, Y_ZIG, DY_STRAND,
    OUT_AUTO_NAME, OUT_FIXED_NAME,
    ALL_DIR_AUTO, ALL_DIR_FIXED,
    FIGSIZE, DPI,
    FIX_PAD_X, FIX_PAD_Y,
)

from mf_lattice_postproc import run_lattice_postprocessing



def main():
    base_dir_abs = os.path.join(PROJECT_ROOT, BASE_DIR)

    run_lattice_postprocessing(
        base_dir=base_dir_abs,
        n_min=N_MIN,
        n_max=N_MAX,
        lattice_max_n=LATTICE_MAX_N,
        flip_width_order=FLIP_WIDTH_ORDER,
        n_repeat=N_REPEAT,
        dx_step=DX_STEP,
        y_zig=Y_ZIG,
        dy_strand=DY_STRAND,
        out_auto_name=OUT_AUTO_NAME,
        out_fixed_name=OUT_FIXED_NAME,
        all_dir_auto=ALL_DIR_AUTO,
        all_dir_fixed=ALL_DIR_FIXED,
        figsize=FIGSIZE,
        dpi=DPI,
        fix_pad_x=FIX_PAD_X,
        fix_pad_y=FIX_PAD_Y,
    )


if __name__ == "__main__":
    main()
