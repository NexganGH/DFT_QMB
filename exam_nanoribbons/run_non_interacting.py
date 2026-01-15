# run_non_interacting.py
import os
import sys


def add_to_syspath(path: str) -> None:
    """Prepend path to sys.path if not already present."""
    if path not in sys.path:
        sys.path.insert(0, path)


def get_project_root() -> str:
    """
    Determine project root robustly.

    - If this file is in PROJECT_ROOT/, returns PROJECT_ROOT
    - If this file is in PROJECT_ROOT/src/, returns PROJECT_ROOT
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # If we're inside a "src" folder, project root is the parent.
    if os.path.basename(this_dir) == "src":
        return os.path.abspath(os.path.join(this_dir, ".."))

    # Otherwise assume this file lives in the project root.
    return this_dir


PROJECT_ROOT = get_project_root()
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")

# Make imports work without installing a package
add_to_syspath(SRC_DIR)
add_to_syspath(INPUT_DIR)

# --- imports (now that sys.path is set) ---
from inputs_non_interacting import (
    MODE, N_SINGLE, N_MIN, N_MAX,
    t, a, nk, eta, n_E, mu,
    base_dir,
)

from zgnr_noninteracting import run_single_case, run_sweep


def main():
    # Force base_dir to be relative to PROJECT_ROOT (not to src/)
    base_dir_abs = os.path.abspath(os.path.join(PROJECT_ROOT, base_dir))

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
        print("\nDONE sweep run.")
        print("Saved under:", base_dir_abs)

    else:
        raise ValueError(f"Unknown MODE={MODE!r}. Use 'single' or 'sweep'.")


if __name__ == "__main__":
    main()
