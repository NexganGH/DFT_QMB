# run_non_interacting.py
import os
import sys

# --- allow importing from ./src without installing a package ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")
if INPUT_DIR not in sys.path:
    sys.path.insert(0, INPUT_DIR)



from inputs_non_interacting import (
    MODE, N_SINGLE, N_MIN, N_MAX,
    t, a, nk, eta, n_E, mu,
    base_dir,
)

from inputs_non_interacting import (
    MODE, N_SINGLE, N_MIN, N_MAX,
    t, a, nk, eta, n_E, mu,
    base_dir,
)

from zgnr_noninteracting import run_single_case, run_sweep


def main():
    # Force base_dir to be relative to project root (NOT src)
    base_dir_abs = os.path.join(PROJECT_ROOT, base_dir)

    if MODE == "single":
        out = run_single_case(
            N_SINGLE,
            t=t, a=a, nk=nk, eta=eta, n_E=n_E, mu=mu,
            base_dir=base_dir_abs,
            copy_to_all=True,
        )
        print("\nDONE single run.")
        print("Saved:", out)

    elif MODE == "sweep":
        run_sweep(
            N_MIN, N_MAX,
            t=t, a=a, nk=nk, eta=eta, n_E=n_E, mu=mu,
            base_dir=base_dir_abs,
        )
    else:
        raise ValueError(f"Unknown MODE={MODE!r}. Use 'single' or 'sweep'.")


if __name__ == "__main__":
    main()
