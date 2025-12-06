# 02_bands_dos_zgnr.py

import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW

from config_zgnr import (
    LABEL_BASE,
    NK_PATH,
    NBANDS,
    ETA_DOS,
)


def bands_and_dos_from_gpw(gpw_file):
    """
    Non-selfconsistent band structure + DOS from a converged gpw.
    Saves an npz and a png.
    """
    calc = GPAW(gpw_file)
    atoms = calc.get_atoms()

    # === GEOMETRY DIAGNOSTIC ===
    print("=== GEOMETRY DIAGNOSTIC ===")
    print("Cell (Å):")
    print(atoms.cell)
    print("PBC flags:", atoms.pbc)

    pos = atoms.get_positions()
    span = pos.max(axis=0) - pos.min(axis=0)
    print("Span (Å) along x, y, z:", span)
    print("===========================\n")
    # ===========================

    # Periodic / ribbon direction = x
    a_dft = atoms.cell.lengths()[0]

    # Path Γ→X along x
    path = atoms.cell.bandpath("GX", npoints=NK_PATH)
    kpts = path.kpts

    bs_calc = calc.fixed_density(
        kpts=kpts,
        symmetry="off",
        nbands=NBANDS,
        txt=f"{LABEL_BASE}_bands.txt",
    )

    ef = bs_calc.get_fermi_level()
    nk = len(kpts)

    E_all = np.zeros((nk, NBANDS), dtype=float)
    for ik in range(nk):
        eigs = bs_calc.get_eigenvalues(kpt=ik)
        n_here = min(len(eigs), NBANDS)
        E_all[ik, :n_here] = eigs[:n_here]

    E_rel_all = E_all - ef

    # Dimensionless k*a grid [0, π]
    k_dimless = np.linspace(0.0, np.pi, nk)

    # DOS with Gaussian broadening
    E_flat = E_rel_all.ravel()
    Emin, Emax = E_flat.min(), E_flat.max()
    pad = 0.2 * (Emax - Emin)
    Emin -= pad
    Emax += pad
    n_E = 2000
    E_grid = np.linspace(Emin, Emax, n_E)

    x = E_grid[:, None] - E_flat[None, :]
    gaussians = np.exp(-0.5 * (x / ETA_DOS) ** 2) / (
        np.sqrt(2 * np.pi) * ETA_DOS
    )
    DOS = gaussians.sum(axis=1) / nk

    # Save all data
    np.savez(
        f"{LABEL_BASE}_bands_dos.npz",
        k_dimless=k_dimless,
        E_rel_all=E_rel_all,
        E_grid=E_grid,
        DOS=DOS,
        a_dft=a_dft,
        ef=ef,
    )

    # Plot and save PNG
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [2, 1]}
    )

    for n in range(NBANDS):
        ax1.plot(k_dimless / np.pi, E_rel_all[:, n], lw=0.5)
    ax1.axhline(0.0, ls="--", lw=0.5, color="k")
    ax1.set_xlabel(r"$k a / \pi$")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("DFT bands (non-magnetic)")

    ax2.plot(DOS, E_grid, lw=1.0)
    ax2.axhline(0.0, ls="--", lw=0.5, color="k")
    ax2.set_xlabel("DOS (states / eV / cell)")
    ax2.set_ylabel("Energy (eV)")
    ax2.set_title("DFT DOS")

    plt.tight_layout()
    plt.savefig(f"{LABEL_BASE}_bands_dos.png", dpi=200)
    plt.close(fig)

    print(f"Bands + DOS stored in {LABEL_BASE}_bands_dos.npz")
    return k_dimless, E_rel_all, a_dft


if __name__ == "__main__":
    gpw_file = f"{LABEL_BASE}_relaxed.gpw"
    bands_and_dos_from_gpw(gpw_file)
