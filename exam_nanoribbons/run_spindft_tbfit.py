# run_spindft_tbfit.py
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_tbfit import (
    BASE_OUTDIR,
    NK_TB_FACTOR, FILLING, MAX_ITER, MIX, TOL,
    USE_SCIPY_IF_AVAILABLE, T0, U0, T_GRID, U_GRID
)
from inputs_spindft import MODE, NY_SINGLE, NY_LIST, M

from spindft_paths import case_dir, ensure_dirs, relabel_cfg_files

from spindft_tb_fit_core import (
    load_dft_central_bands,
    compute_tb_central_bands,
    band_cost,
    save_tbfit_npz,
)

from spindft_tb_fit_plots import (
    plot_pi_fit_pretty,
    plot_magnetization_profile_pretty,
)


def _try_minimize():
    try:
        from scipy.optimize import minimize
        return minimize
    except Exception:
        return None


def fit_one_case(mode: str, Ny: int):
    case = case_dir(BASE_OUTDIR, mode, Ny)
    dirs = ensure_dirs(case)
    files = relabel_cfg_files(dirs, Ny=Ny, M=M)

    if not os.path.isfile(files["bands_pi_npz"]):
        print(f"[SKIP] missing bands_pi_npz for Ny={Ny}: {files['bands_pi_npz']}")
        return
    if not os.path.isfile(files["mag_AB_npz"]):
        print(f"[SKIP] missing mag_AB_npz for Ny={Ny}: {files['mag_AB_npz']}")
        return

    dft = load_dft_central_bands(files["bands_pi_npz"])
    Nk_dft = len(dft["k_plot"])
    Nk_tb = int(NK_TB_FACTOR) * Nk_dft

    x0 = np.array([T0, U0], float)
    minimize = _try_minimize() if USE_SCIPY_IF_AVAILABLE else None

    if minimize is not None:
        res = minimize(
            lambda x: band_cost(x, dft, Ny, Nk_tb, FILLING, MAX_ITER, MIX, TOL),
            x0,
            method="Nelder-Mead",
            options={"maxiter": 80, "disp": True},
        )
        t_opt, U_opt = res.x
        cost = float(res.fun)
    else:
        best = 1e99
        t_opt, U_opt = float(T0), float(U0)
        for t in T_GRID:
            for U in U_GRID:
                c = band_cost((t, U), dft, Ny, Nk_tb, FILLING, MAX_ITER, MIX, TOL)
                if c < best:
                    best = c
                    t_opt, U_opt = float(t), float(U)
        cost = float(best)

    tb = compute_tb_central_bands(
        Ny, t_opt, U_opt, Nk_tb, dft["k_plot"],
        FILLING, MAX_ITER, MIX, TOL
    )
    result_tb = tb["result"]

    mag = np.load(files["mag_AB_npz"], allow_pickle=True)
    mA_dft = np.array(mag["mA"], float).ravel()
    mB_dft = np.array(mag["mB"], float).ravel()

    mA_tb = np.array(result_tb["mA"], float).ravel()
    mB_tb = np.array(result_tb["mB"], float).ravel()

    # pretty plots
    plot_pi_fit_pretty(
        dft["k_plot"], dft["E_val_up"], dft["E_cond_up"],
        tb["E_val_up_tb"], tb["E_cond_up_tb"],
        Ny=Ny, t=t_opt, U=U_opt, out_png=files["tbfit_bands_png"],
    )
    plot_magnetization_profile_pretty(
        mA_dft, mB_dft, mA_tb, mB_tb,
        Ny=Ny, t=t_opt, U=U_opt, out_png=files["tbfit_mag_png"],
    )

    # save npz with the keys your postproc expects
    save_tbfit_npz(
        files["tbfit_npz"],
        Ny=Ny, M=M,
        t=t_opt, U=U_opt,
        cost=cost,
        k_plot=dft["k_plot"],
        E_val_dft=dft["E_val_up"],
        E_cond_dft=dft["E_cond_up"],
        E_val_tb=tb["E_val_up_tb"],
        E_cond_tb=tb["E_cond_up_tb"],
        mA_dft=mA_dft, mB_dft=mB_dft,
        mA_tb=mA_tb, mB_tb=mB_tb,
    )

    print(f"[OK] TB fit Ny={Ny}: t={t_opt:.4f} U={U_opt:.4f} cost={cost:.3e}")


def main():
    if MODE == "single":
        fit_one_case("single", NY_SINGLE)
    elif MODE == "sweep_N":
        for Ny in NY_LIST:
            fit_one_case("sweep_N", Ny)
    else:
        raise ValueError("MODE must be 'single' or 'sweep_N'")


if __name__ == "__main__":
    main()
