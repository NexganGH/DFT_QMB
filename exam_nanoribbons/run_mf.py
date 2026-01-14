# run_mf.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")
if INPUT_DIR not in sys.path:
    sys.path.insert(0, INPUT_DIR)

from inputs_mf import (
    MODE,
    N_SINGLE, U_SINGLE,
    N_MIN, N_MAX, U_FIXED_FOR_SWEEP_N,
    N_FIXED_FOR_SWEEP_U, Umin_over_t, Umax_over_t, dU_over_t,
    t, a,
    Nk, filling, max_iter, mix, tol, m0, verbose,
    dos_nE, dos_eta,
    base_dir,
)

from zgnr_mf import run_single_case, run_sweep_N, run_sweep_U



def main():
    base_dir_abs = os.path.join(PROJECT_ROOT, base_dir)

    if MODE == "single":
        out = run_single_case(
            N=N_SINGLE, t=t, U=U_SINGLE, a=a,
            Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir_abs, mode_tag="single",
            copy_to_all=True, verbose=verbose,
        )
        print("\nDONE single.")
        print("Saved:", out)

    elif MODE == "sweep_N":
        run_sweep_N(
            N_min=N_MIN, N_max=N_MAX,
            t=t, U=U_FIXED_FOR_SWEEP_N,
            a=a, Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir_abs, mode_tag="sweep_N",
            verbose=verbose,
        )

    elif MODE == "sweep_U":
        run_sweep_U(
            N_fixed=N_FIXED_FOR_SWEEP_U,
            t=t,
            Umin_over_t=Umin_over_t, Umax_over_t=Umax_over_t, dU_over_t=dU_over_t,
            a=a, Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir_abs, mode_tag="sweep_U",
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown MODE={MODE!r}. Use 'single', 'sweep_N', or 'sweep_U'.")


if __name__ == "__main__":
    main()
