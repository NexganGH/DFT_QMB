# 04_fit_tb_to_dft.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from config_zgnr import LABEL_BASE, N_ZIGZAG, get_tb_module


def tb_bands_on_kgrid(k_dimless, N_zigzag, t_hop, a_dft, H_zgnr_k):
    """
    Compute TB bands on same k-grid used for DFT.
    """
    Nk = len(k_dimless)
    dim = 2 * N_zigzag
    E_tb = np.zeros((Nk, dim), dtype=float)

    # dimensionless k*a -> k (1/Å) using same a
    ks = k_dimless / a_dft

    for i, k in enumerate(ks):
        Hk = H_zgnr_k(k, N_zigzag, t=t_hop, a=a_dft)
        w, _ = np.linalg.eigh(Hk)
        E_tb[i, :] = np.sort(w.real)
    return E_tb


def fit_tb_to_dft(k_dimless, E_pi_dft, N_zigzag, a_dft, H_zgnr_k,
                  t_initial=-2.7, dE_initial=0.0):
    """
    Fit TB (t, ΔE) to DFT π-bands by minimising mean squared error.
    """

    def loss(params):
        t_hop, dE = params
        E_tb = tb_bands_on_kgrid(k_dimless, N_zigzag, t_hop, a_dft, H_zgnr_k)
        E_tb_shift = E_tb + dE
        diff = E_pi_dft - E_tb_shift
        return np.mean(diff ** 2)

    x0 = np.array([t_initial, dE_initial])
    res = minimize(loss, x0, method="Nelder-Mead")
    t_opt, dE_opt = res.x
    rmse = np.sqrt(res.fun)

    E_tb_opt = tb_bands_on_kgrid(k_dimless, N_zigzag, t_opt, a_dft, H_zgnr_k) + dE_opt

    return {
        "t": t_opt,
        "dE": dE_opt,
        "rmse": rmse,
        "E_tb": E_tb_opt,
    }


def main():
    # 1. load π-bands
    data = np.load(f"{LABEL_BASE}_pi_bands.npz")
    k_dimless = data["k_dimless"]
    E_pi_dft = data["E_pi_dft"]
    a_dft = data["a_dft"]
    band_indices = data["band_indices"]
    print(f"Using π-bands with indices: {band_indices}")

    # 2. import TB module and get H_zgnr_k
    tb_mod = get_tb_module()
    H_zgnr_k = tb_mod.H_zgnr_k

    # 3. fit
    fit = fit_tb_to_dft(
        k_dimless, E_pi_dft, N_ZIGZAG, a_dft, H_zgnr_k,
        t_initial=-2.7, dE_initial=0.0
    )

    print("\n=== TB fit results ===")
    print(f"t   = {fit['t']:.4f} eV")
    print(f"ΔE  = {fit['dE']:.4f} eV (global shift)")
    print(f"RMSE= {fit['rmse']:.4f} eV")

    # 4. save fit results
    np.savez(
        f"{LABEL_BASE}_tb_fit_results.npz",
        t=fit["t"],
        dE=fit["dE"],
        rmse=fit["rmse"],
        k_dimless=k_dimless,
        E_tb=fit["E_tb"],
        E_pi_dft=E_pi_dft,
        a_dft=a_dft,
    )

    # 5. plot comparison
    E_tb_opt = fit["E_tb"]

    fig, ax = plt.subplots(figsize=(5, 4))
    # DFT π-bands
    for n in range(E_pi_dft.shape[1]):
        ax.plot(k_dimless / np.pi, E_pi_dft[:, n],
                lw=1.0, color="C0", alpha=0.7)
    # TB bands
    for n in range(E_tb_opt.shape[1]):
        ax.plot(k_dimless / np.pi, E_tb_opt[:, n],
                lw=1.0, ls="--", color="C1", alpha=0.9)

    ax.axhline(0.0, ls="--", lw=0.5, color="k")
    ax.set_xlabel(r"$k a / \pi$")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"ZGNR-{N_ZIGZAG}: DFT π vs TB fit")

    plt.tight_layout()
    plt.savefig(f"{LABEL_BASE}_dft_vs_tb.png", dpi=200)
    plt.close(fig)

    print(f"Fit and comparison plot stored as {LABEL_BASE}_tb_fit_results.npz "
          f"and {LABEL_BASE}_dft_vs_tb.png")


if __name__ == "__main__":
    main()
