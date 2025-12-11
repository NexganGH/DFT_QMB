# 05_fit.py
#
# Fit t and U of the mean-field TB model to the DFT CENTRAL π bands.
# Uses:
#   - DFT data from cfg.bands_pi_npz (produced by 04_zgnr_bands_pi.py)
#   - DFT magnetization profile from cfg.mag_AB_npz (03_zgnr_magnetization.py)
#   - MF TB solver solve_zgnr_mf from mft/magnetisation_mft.py
#
# Output:
#   - cfg.tb_fit_npz : full fit info (t, U, bands, magnetizations, etc.)
#   - cfg.tb_fit_bands_png : DFT vs TB central bands overlay
#   - cfg.tb_fit_mag_png   : DFT vs TB magnetization profile overlay

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import config_zgnr_LCAO as cfg

# --- Try to import SciPy minimizer (for convenience) ---
try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("WARNING: SciPy not found, will fall back to coarse grid search.")

# --- Import MF solver from your TB code ---
# Assumes folder 'mft' is a sibling directory.
here = os.path.dirname(os.path.abspath(__file__))
mft_path = os.path.join(here, "mft")
if mft_path not in sys.path:
    sys.path.append(mft_path)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import config_zgnr_LCAO as cfg

# --------------------------------------------------------
# Add parent directory to sys.path so we can import mft/
# --------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
parent = os.path.abspath(os.path.join(here, ".."))   # <- go one level UP

if parent not in sys.path:
    sys.path.insert(0, parent)

# Now import from mft/
from mft.magnetisation_mft import solve_zgnr_mf


# ----------------------------------------------------------
# Helpers: central-band selection, etc.
# ----------------------------------------------------------

def find_central_bands_for_spin(E_rel_spin, band_list):
    """
    Given E_rel_spin[k, n] and a list of candidate bands 'band_list',
    return indices of central valence and conduction bands, plus
    their dispersions as arrays E_val[k], E_cond[k].
    """
    band_list = np.array(band_list, dtype=int)
    if band_list.size == 0:
        return None, None, None, None

    # Mean energy for each candidate
    E_mean = E_rel_spin[:, band_list].mean(axis=0)

    # Valence: <0, closest to 0
    mask_val = (E_mean < 0.0)
    if np.any(mask_val):
        idx = np.argmin(np.abs(E_mean[mask_val]))
        val_band = band_list[mask_val][idx]
    else:
        # Fallback: closest to 0
        idx = np.argmin(np.abs(E_mean))
        val_band = band_list[idx]

    # Conduction: >0, closest to 0
    mask_cond = (E_mean > 0.0)
    if np.any(mask_cond):
        idx = np.argmin(np.abs(E_mean[mask_cond]))
        cond_band = band_list[mask_cond][idx]
    else:
        # Fallback: next-closest
        order = np.argsort(np.abs(E_mean))
        cond_band = band_list[order[0]] if band_list.size == 1 else band_list[order[1]]

    E_val = E_rel_spin[:, val_band]
    E_cond = E_rel_spin[:, cond_band]
    return val_band, cond_band, E_val, E_cond


def load_dft_central_bands():
    """Load DFT band data and extract central valence/conduction for each spin."""
    data = np.load(cfg.bands_pi_npz, allow_pickle=True)

    # k-axis (dimensionless ka/π in [0,1])
    if "k_plot" in data:
        k_plot = data["k_plot"]
    else:
        k_plot = data["k_dimless"] / np.pi

    E_rel = data["E_rel"]         # shape (nspins, nkpts, nbands)
    bands_up = data["bands_up"]
    bands_dn = data["bands_dn"]
    efermi = float(data["efermi"])

    nspins, nkpts, nbands = E_rel.shape
    print("Loaded DFT bands from:", cfg.bands_pi_npz)
    print("nspins =", nspins, "nkpts =", nkpts, "nbands =", nbands)
    print("π bands (up):", bands_up)
    print("π bands (dn):", bands_dn)

    # Spin up central bands
    val_up_idx, cond_up_idx, E_val_up, E_cond_up = find_central_bands_for_spin(
        E_rel[0], bands_up
    )

    # Spin down (if present)
    if nspins > 1 and bands_dn.size > 0:
        val_dn_idx, cond_dn_idx, E_val_dn, E_cond_dn = find_central_bands_for_spin(
            E_rel[1], bands_dn
        )
    else:
        val_dn_idx = cond_dn_idx = None
        E_val_dn = E_cond_dn = None

    dft = {
        "k_plot": k_plot,
        "E_val_up": E_val_up,
        "E_cond_up": E_cond_up,
        "E_val_dn": E_val_dn,
        "E_cond_dn": E_cond_dn,
        "val_up_idx": val_up_idx,
        "cond_up_idx": cond_up_idx,
        "val_dn_idx": val_dn_idx,
        "cond_dn_idx": cond_dn_idx,
        "efermi": efermi,
    }
    return dft


def compute_tb_central_bands(Ny, t, U, Nk_tb, k_plot_target):
    """
    Run MF TB solver with (t,U) and return central bands interpolated
    on the same k_plot grid as DFT (ka/π in [0,1]).

    We use solve_zgnr_mf with k in [-π, π], then keep k>=0 half and
    interpolate onto the target grid.
    """
    # Run MF solver (a=1.0, so k-grid in [-π,π])
    result = solve_zgnr_mf(
        Ny=Ny,
        U=U,
        t=t,
        a=1.0,
        Nk=Nk_tb,
        filling=1.0,
        max_iter=200,
        mix=0.1,
        tol=1e-5,
        verbose=False,
    )

    k_grid = result["k_grid"]        # [-π, π)
    E = result["E"]                  # (2, Nk, 2*Ny)
    mu = result["mu"]

    # Use energies relative to mu (like DFT relative to EF)
    E_rel_tb = E - mu

    # Keep only k>=0 part (Gamma->Z)
    mask_pos = (k_grid >= 0.0)
    k_pos = k_grid[mask_pos]               # in [0, π]
    k_plot_tb = k_pos / np.pi             # ka/π since a=1

    # Central bands per spin (on TB grid)
    bands_pi_tb_up = list(range(E_rel_tb.shape[2]))  # all bands potential π; we'll re-select
    val_up_idx, cond_up_idx, E_val_up_tb, E_cond_up_tb = find_central_bands_for_spin(
        E_rel_tb[0][mask_pos, :], bands_pi_tb_up
    )

    # Spin down
    val_dn_idx = cond_dn_idx = None
    E_val_dn_tb = E_cond_dn_tb = None

    if E_rel_tb.shape[0] > 1:
        bands_pi_tb_dn = list(range(E_rel_tb.shape[2]))
        val_dn_idx, cond_dn_idx, E_val_dn_tb, E_cond_dn_tb = find_central_bands_for_spin(
            E_rel_tb[1][mask_pos, :], bands_pi_tb_dn
        )

    # Interpolate onto target k_plot grid (DFT)
    def interp_or_none(k_src, E_src, k_dst):
        if E_src is None:
            return None
        return np.interp(k_dst, k_src, E_src)

    E_val_up_interp = interp_or_none(k_plot_tb, E_val_up_tb, k_plot_target)
    E_cond_up_interp = interp_or_none(k_plot_tb, E_cond_up_tb, k_plot_target)
    E_val_dn_interp = interp_or_none(k_plot_tb, E_val_dn_tb, k_plot_target)
    E_cond_dn_interp = interp_or_none(k_plot_tb, E_cond_dn_tb, k_plot_target)

    tb = {
        "k_plot_tb": k_plot_tb,
        "E_val_up_tb": E_val_up_interp,
        "E_cond_up_tb": E_cond_up_interp,
        "E_val_dn_tb": E_val_dn_interp,
        "E_cond_dn_tb": E_cond_dn_interp,
        "result": result,
        "t": t,
        "U": U,
    }
    return tb


def band_fit_cost(params, dft, Ny, Nk_tb):
    """Cost function for fitting (t, U) -> central bands."""
    t, U = params
    if t <= 0 or U <= 0:
        return 1e6

    tb = compute_tb_central_bands(Ny, t, U, Nk_tb, dft["k_plot"])

    cost = 0.0
    # Spin up
    cost += np.mean((tb["E_val_up_tb"] - dft["E_val_up"])**2)
    cost += np.mean((tb["E_cond_up_tb"] - dft["E_cond_up"])**2)

    # Spin down if present
    if dft["E_val_dn"] is not None and tb["E_val_dn_tb"] is not None:
        cost += np.mean((tb["E_val_dn_tb"] - dft["E_val_dn"])**2)
        cost += np.mean((tb["E_cond_dn_tb"] - dft["E_cond_dn"])**2)

    return float(cost)


# ----------------------------------------------------------
# Main: perform the fit and save results
# ----------------------------------------------------------

def main():
    Ny = cfg.Ny
    # 1. Load DFT central bands
    dft = load_dft_central_bands()

    # 2. Set TB k-grid size (use same number of k-points *2* as DFT)
    Nk_dft = len(dft["k_plot"])
    Nk_tb = 2 * Nk_dft

    # 3. Initial guesses for t and U (in eV)
    t0 = 2.7
    U0 = 2.0
    x0 = np.array([t0, U0])

    print("\nStarting fit of (t, U) to central DFT bands...")
    print("Initial guess: t0 =", t0, "eV,  U0 =", U0, "eV")

    if HAVE_SCIPY:
        res = minimize(
            band_fit_cost,
            x0,
            args=(dft, Ny, Nk_tb),
            method="Nelder-Mead",
            options={"maxiter": 80, "disp": True}
        )
        t_opt, U_opt = res.x
        print("\nFit finished (SciPy Nelder–Mead).")
        print("t_opt =", t_opt, "eV")
        print("U_opt =", U_opt, "eV")
        print("Final cost =", res.fun)
    else:
        # coarse grid search as a fallback
        t_vals = np.linspace(2.0, 3.2, 6)
        U_vals = np.linspace(1.0, 3.0, 6)
        best_cost = 1e9
        t_opt = t0
        U_opt = U0
        for t in t_vals:
            for U in U_vals:
                c = band_fit_cost((t, U), dft, Ny, Nk_tb)
                if c < best_cost:
                    best_cost = c
                    t_opt, U_opt = t, U
                    print(f"New best: t={t:.3f}, U={U:.3f}, cost={c:.4e}")
        print("\nGrid-search finished.")
        print("t_opt =", t_opt, "eV")
        print("U_opt =", U_opt, "eV")
        print("Best cost =", best_cost)

    # 4. Compute TB bands and magnetization for optimal (t,U)
    tb_best = compute_tb_central_bands(Ny, t_opt, U_opt, Nk_tb, dft["k_plot"])
    result_tb = tb_best["result"]  # full MF result with mA, mB, etc.

    # 5. Load DFT magnetization profile (per strand) for comparison
    mag_dft = np.load(cfg.mag_AB_npz, allow_pickle=True)
    m_indices = mag_dft["m_indices"]
    mA_dft = mag_dft["mA"]
    mB_dft = mag_dft["mB"]

    mA_tb = result_tb["mA"]
    mB_tb = result_tb["mB"]
    strands_tb = np.arange(len(mA_tb))

    # 6. Plot: central bands DFT vs TB
    k = dft["k_plot"]
    plt.figure(figsize=(6, 5))

    # DFT
    plt.plot(k, dft["E_val_up"], "k-",  lw=2.0, label="DFT val (↑)")
    plt.plot(k, dft["E_cond_up"], "k--", lw=2.0, label="DFT cond (↑)")
    if dft["E_val_dn"] is not None:
        plt.plot(k, dft["E_val_dn"], color="gray", linestyle="-", lw=2.0, label="DFT val (↓)")
        plt.plot(k, dft["E_cond_dn"], color="gray", linestyle="--", lw=2.0, label="DFT cond (↓)")


    # TB
    plt.plot(k, tb_best["E_val_up_tb"], "C0-",  lw=1.5, label="TB val (↑)")
    plt.plot(k, tb_best["E_cond_up_tb"], "C1-",  lw=1.5, label="TB cond (↑)")
    if tb_best["E_val_dn_tb"] is not None:
        plt.plot(k, tb_best["E_val_dn_tb"], "C0--", lw=1.5, label="TB val (↓)")
        plt.plot(k, tb_best["E_cond_dn_tb"], "C1--", lw=1.5, label="TB cond (↓)")

    plt.axhline(0.0, ls="--", lw=0.8, color="k")
    plt.xlabel(r"$k a / \pi$ (0 → $\Gamma$, 1 → Z)")
    plt.ylabel(r"$E - E_F$ (eV)")
    plt.title(f"Central π bands: DFT vs TB  (t={t_opt:.2f} eV, U={U_opt:.2f} eV)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.tb_fit_bands_png, dpi=300)
    plt.close()
    print("Saved band-fit overlay to", cfg.tb_fit_bands_png)

    # 7. Plot: magnetization profile DFT vs TB
    plt.figure(figsize=(6, 4))
    plt.plot(m_indices, mA_dft, "ko-", label="DFT mA")
    plt.plot(m_indices, mB_dft, "ks--", label="DFT mB")

    plt.plot(strands_tb, mA_tb, "C0o-", label="TB mA")
    plt.plot(strands_tb, mB_tb, "C1s--", label="TB mB")

    plt.axhline(0.0, ls="--", lw=0.8, color="k")
    plt.xlabel("Strand index m")
    plt.ylabel("Magnetization (μB)")
    plt.title("Magnetization profile: DFT vs TB fit")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.tb_fit_mag_png, dpi=300)
    plt.close()
    print("Saved magnetization overlay to", cfg.tb_fit_mag_png)

    # 8. Save everything for later post-processing
    np.savez(
        cfg.tb_fit_npz,
        Ny=Ny,
        t_opt=t_opt,
        U_opt=U_opt,
        k_plot=dft["k_plot"],
        dft=dft,
        tb_bands=tb_best,
        mA_dft=mA_dft,
        mB_dft=mB_dft,
        mA_tb=mA_tb,
        mB_tb=mB_tb,
    )
    print("Saved fit data to", cfg.tb_fit_npz)


if __name__ == "__main__":
    main()
