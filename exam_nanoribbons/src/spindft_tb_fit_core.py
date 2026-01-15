# src/spindft_tb_fit_core.py
import numpy as np

# Import your MF solver (this must be in src/ and importable)
# It must return a dict with keys: k_grid, E, mu, mA, mB
from zgnr_mf import solve_zgnr_mf


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _safe_get(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def find_central_bands_for_spin(E_rel_spin, band_list):
    """
    Given E_rel_spin[k, n] and a list of candidate band indices,
    return the central valence band (closest below 0) and
    central conduction band (closest above 0) as curves vs k.
    """
    band_list = np.array(band_list, dtype=int)
    if band_list.size == 0:
        return None, None, None, None

    E_mean = E_rel_spin[:, band_list].mean(axis=0)

    # valence: mean < 0 and closest to 0
    mask_val = (E_mean < 0.0)
    if np.any(mask_val):
        idx = np.argmin(np.abs(E_mean[mask_val]))
        val_band = band_list[mask_val][idx]
    else:
        val_band = band_list[np.argmin(np.abs(E_mean))]

    # conduction: mean > 0 and closest to 0
    mask_cond = (E_mean > 0.0)
    if np.any(mask_cond):
        idx = np.argmin(np.abs(E_mean[mask_cond]))
        cond_band = band_list[mask_cond][idx]
    else:
        order = np.argsort(np.abs(E_mean))
        cond_band = band_list[order[0]] if band_list.size == 1 else band_list[order[1]]

    return val_band, cond_band, E_rel_spin[:, val_band], E_rel_spin[:, cond_band]


# ------------------------------------------------------------
# Load DFT central bands from your run_spindft bands_pi_npz
# ------------------------------------------------------------
def load_dft_central_bands(bands_pi_npz: str):
    """
    Reads the npz produced by spindft_bands_core.compute_bands_pi().

    Expected keys:
      - k_plot
      - E_rel (nspins, Nk, nbands)
      - bands_up, bands_dn (lists of pi-like band indices)
    """
    data = np.load(bands_pi_npz, allow_pickle=True)

    k_plot = np.asarray(data["k_plot"], float).ravel()
    E_rel = np.asarray(data["E_rel"], float)
    bands_up = np.asarray(data["bands_up"], int)
    bands_dn = np.asarray(data["bands_dn"], int) if "bands_dn" in data else np.array([], int)

    # Spin up central bands
    _, _, Eval_up, Econd_up = find_central_bands_for_spin(E_rel[0], bands_up)

    # Spin down (if present)
    Eval_dn = None
    Econd_dn = None
    if E_rel.shape[0] > 1 and bands_dn.size > 0:
        _, _, Eval_dn, Econd_dn = find_central_bands_for_spin(E_rel[1], bands_dn)

    return {
        "k_plot": k_plot,
        "E_val_up": Eval_up,
        "E_cond_up": Econd_up,
        "E_val_dn": Eval_dn,
        "E_cond_dn": Econd_dn,
    }


# ------------------------------------------------------------
# Compute TB central bands on same k grid as DFT
# ------------------------------------------------------------
def compute_tb_central_bands(
    Ny: int,
    t: float,
    U: float,
    Nk_tb: int,
    k_plot_target,
    filling: float,
    max_iter: int,
    mix: float,
    tol: float,
):
    """
    Runs TB mean-field solver and returns central val/conduction bands
    interpolated onto the DFT k_plot grid.
    """
    k_plot_target = np.asarray(k_plot_target, float).ravel()

    result = solve_zgnr_mf(
        Ny=Ny,
        U=U,
        t=t,
        a=1.0,
        Nk=Nk_tb,
        filling=filling,
        max_iter=max_iter,
        mix=mix,
        tol=tol,
        verbose=False,
    )

    k_grid = np.asarray(result["k_grid"], float).ravel()  # expected in [-pi, pi) or similar
    E = np.asarray(result["E"], float)                    # (nspins, Nk, 2*Ny)
    mu = float(result["mu"])

    # energies relative to mu (like DFT E-Ef)
    E_rel_tb = E - mu

    # keep only k>=0 part (Gamma->Z)
    mask_pos = (k_grid >= 0.0)
    k_pos = k_grid[mask_pos]
    k_plot_tb = k_pos / np.pi  # since a=1 => ka/pi in [0,1]

    all_bands = list(range(E_rel_tb.shape[2]))

    # spin up
    _, _, Eval_up_tb, Econd_up_tb = find_central_bands_for_spin(
        E_rel_tb[0][mask_pos, :], all_bands
    )

    # spin down (optional)
    Eval_dn_tb = None
    Econd_dn_tb = None
    if E_rel_tb.shape[0] > 1:
        _, _, Eval_dn_tb, Econd_dn_tb = find_central_bands_for_spin(
            E_rel_tb[1][mask_pos, :], all_bands
        )

    def _interp(y):
        if y is None:
            return None
        return np.interp(k_plot_target, k_plot_tb, np.asarray(y, float).ravel())

    return {
        "E_val_up_tb": _interp(Eval_up_tb),
        "E_cond_up_tb": _interp(Econd_up_tb),
        "E_val_dn_tb": _interp(Eval_dn_tb),
        "E_cond_dn_tb": _interp(Econd_dn_tb),
        "result": result,
    }


# ------------------------------------------------------------
# Cost function for fitting (t,U) to DFT central bands
# ------------------------------------------------------------
def band_cost(params, dft, Ny, Nk_tb, filling, max_iter, mix, tol):
    t, U = float(params[0]), float(params[1])
    if t <= 0.0 or U <= 0.0:
        return 1e9

    tb = compute_tb_central_bands(Ny, t, U, Nk_tb, dft["k_plot"], filling, max_iter, mix, tol)

    cost = 0.0
    cost += np.mean((tb["E_val_up_tb"] - dft["E_val_up"]) ** 2)
    cost += np.mean((tb["E_cond_up_tb"] - dft["E_cond_up"]) ** 2)

    if dft["E_val_dn"] is not None and tb["E_val_dn_tb"] is not None:
        cost += np.mean((tb["E_val_dn_tb"] - dft["E_val_dn"]) ** 2)
        cost += np.mean((tb["E_cond_dn_tb"] - dft["E_cond_dn"]) ** 2)

    return float(cost)


# ------------------------------------------------------------
# Canonical saver (IMPORTANT: writes t_fit/U_fit expected by plotters)
# ------------------------------------------------------------
def save_tbfit_npz(
    out_npz: str,
    Ny: int,
    M: int,
    t: float,
    U: float,
    cost: float,
    k_plot,
    E_val_dft,
    E_cond_dft,
    E_val_tb,
    E_cond_tb,
    mA_dft,
    mB_dft,
    mA_tb,
    mB_tb,
    **extra,
):
    """
    Saves TB fit outputs with a stable schema.

    Required by your pretty postproc scripts:
      Ny
      t_fit, U_fit
      mA_dft, mB_dft
      mA_tb,  mB_tb

    Also saves aliases:
      t_opt, U_opt
    """
    np.savez(
        out_npz,

        Ny=int(Ny),
        M=int(M),

        # canonical keys expected by plotters
        t_fit=float(t),
        U_fit=float(U),

        # aliases (useful if other scripts expect them)
        t_opt=float(t),
        U_opt=float(U),

        cost=float(cost),

        k_plot=np.asarray(k_plot, float),

        E_val_dft=np.asarray(E_val_dft, float),
        E_cond_dft=np.asarray(E_cond_dft, float),
        E_val_tb=np.asarray(E_val_tb, float),
        E_cond_tb=np.asarray(E_cond_tb, float),

        mA_dft=np.asarray(mA_dft, float),
        mB_dft=np.asarray(mB_dft, float),
        mA_tb=np.asarray(mA_tb, float),
        mB_tb=np.asarray(mB_tb, float),

        **extra,
    )
