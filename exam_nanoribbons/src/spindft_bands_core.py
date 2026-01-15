# spindft_bands_core.py
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW


def compute_bands_pi(
    gpw_path: str,
    npoints_kpath: int,
    nbands: int,
    e_min_pi: float,
    e_max_pi: float,
    bands_full_png: str | None,
    bands_pi_png: str | None,
    bands_pi_npz: str | None,
):
    """
    Compute spin-polarized band structure along periodic z direction (Γ -> Z),
    save full-band plot (optional), π-window plot (optional), and always save .npz (optional).

    The saved .npz contains keys compatible with later post-processing:
      k_frac, k_dimless, k_plot, energies, E_rel, efermi, bands_up, bands_dn
    """
    gs = GPAW(gpw_path)
    efermi = float(gs.get_fermi_level())

    # k along z: Γ -> Z (fractional kz 0 -> 0.5)
    k_frac = np.linspace(0.0, 0.5, int(npoints_kpath))
    kpts = np.zeros((len(k_frac), 3))
    kpts[:, 2] = k_frac

    # fixed density band calculation
    bs = gs.fixed_density(
        nbands=int(nbands),
        kpts=kpts,
        symmetry="off",
        txt=None,
    )

    nspins = int(bs.wfs.nspins)
    Nk = len(k_frac)
    nb = int(nbands)

    energies = np.zeros((nspins, Nk, nb), float)
    for s in range(nspins):
        for ik in range(Nk):
            energies[s, ik, :] = bs.get_eigenvalues(kpt=ik, spin=s)

    E_rel = energies - efermi  # relative to Ef

    # x-axis ka/pi in [0,1]
    k_dimless = 2.0 * np.pi * k_frac   # ka in [0, pi]
    k_plot = k_dimless / np.pi         # ka/pi in [0, 1]

    # -----------------------
    # Full bands plot (optional)
    # -----------------------
    if bands_full_png:
        plt.figure(figsize=(5, 6))
        for s in range(nspins):
            for n in range(nb):
                plt.plot(k_plot, E_rel[s, :, n], lw=0.6)
        plt.axhline(0.0, ls="--", lw=0.9, color="0.35")
        plt.xticks([0.0, 1.0], [r"$\Gamma$", r"$Z$"])
        plt.xlabel(r"$ka/\pi$")
        plt.ylabel(r"$E - E_F$ (eV)")
        plt.tight_layout()
        plt.savefig(bands_full_png, dpi=300)
        plt.close()

    # -----------------------
    # π-like band selection
    # -----------------------
    pi_band_indices = []
    for s in range(nspins):
        mask = (E_rel[s] > e_min_pi) & (E_rel[s] < e_max_pi)
        inds = [n for n in range(nb) if mask[:, n].any()]
        pi_band_indices.append(np.array(inds, dtype=int))

    # -----------------------
    # π bands plot (optional)
    # -----------------------
    if bands_pi_png:
        plt.figure(figsize=(5, 6))
        colors = ["tab:orange", "tab:blue"]
        for s in range(nspins):
            for ii, n in enumerate(pi_band_indices[s]):
                label = None
                if ii == 0:
                    label = "spin up" if s == 0 else "spin down"
                plt.plot(k_plot, E_rel[s, :, n], lw=1.0, color=colors[s % 2], label=label)

        plt.axhline(0.0, ls="--", lw=0.9, color="0.35")
        plt.xticks([0.0, 1.0], [r"$\Gamma$", r"$Z$"])
        plt.ylim(e_min_pi, e_max_pi)
        plt.xlabel(r"$ka/\pi$")
        plt.ylabel(r"$E - E_F$ (eV)")
        if nspins > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(bands_pi_png, dpi=300)
        plt.close()


    if bands_pi_npz:
        np.savez(
            bands_pi_npz,
            k_frac=k_frac,
            k_dimless=k_dimless,
            k_plot=k_plot,
            energies=energies,
            E_rel=E_rel,
            efermi=efermi,
            bands_up=pi_band_indices[0],
            bands_dn=pi_band_indices[1] if nspins > 1 else np.array([], dtype=int),
        )

    return bands_pi_npz
