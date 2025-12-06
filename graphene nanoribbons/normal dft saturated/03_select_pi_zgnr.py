# 03_select_pi_zgnr.py

import numpy as np
import matplotlib.pyplot as plt

from config_zgnr import LABEL_BASE, N_ZIGZAG


def select_pi_bands(E_rel_all, N_zigzag):
    """
    Select 2*N_zigzag π-bands closest to EF:
    bands with smallest average |E| over k.
    """
    Nk, nbands = E_rel_all.shape
    n_pi = 2 * N_zigzag
    if nbands < n_pi:
        raise ValueError(
            f"Not enough bands ({nbands}) to extract 2*N={n_pi} π-bands."
        )

    avg_abs = np.mean(np.abs(E_rel_all), axis=0)
    band_indices = np.argsort(avg_abs)[:n_pi]
    band_indices = np.sort(band_indices)

    E_pi = E_rel_all[:, band_indices]
    E_pi_sorted = np.sort(E_pi, axis=1)
    return E_pi_sorted, band_indices


def main():
    data = np.load(f"{LABEL_BASE}_bands_dos.npz")
    k_dimless = data["k_dimless"]
    E_rel_all = data["E_rel_all"]
    a_dft = data["a_dft"]

    E_pi_dft, band_indices = select_pi_bands(E_rel_all, N_ZIGZAG)

    # save π-bands
    np.savez(
        f"{LABEL_BASE}_pi_bands.npz",
        k_dimless=k_dimless,
        E_pi_dft=E_pi_dft,
        band_indices=band_indices,
        a_dft=a_dft,
    )

    print(f"Selected π-band indices: {band_indices}")
    print(f"Stored in {LABEL_BASE}_pi_bands.npz")

    # optional quick plot
    fig, ax = plt.subplots(figsize=(5, 4))
    for n in range(E_pi_dft.shape[1]):
        ax.plot(k_dimless / np.pi, E_pi_dft[:, n], lw=1.0)

    ax.axhline(0.0, ls="--", lw=0.5, color="k")
    ax.set_xlabel(r"$k a / \pi$")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"DFT π-bands, ZGNR-{N_ZIGZAG}")

    plt.tight_layout()
    plt.savefig(f"{LABEL_BASE}_pi_bands.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
