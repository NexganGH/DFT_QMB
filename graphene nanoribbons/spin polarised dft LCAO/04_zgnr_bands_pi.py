# 04_zgnr_bands_pi.py
#
# Band structure of ZGNR along the periodic z direction.
# Uses explicit k-point list along z (Gamma->BZ boundary),
# no ASE "path strings" like 'GZ'.

from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt
import config_zgnr_LCAO as cfg


def main():
    # --- 1. Restart from converged SCF ---
    gs_calc = GPAW(cfg.gpw_file)
    atoms = gs_calc.get_atoms()

    print("Loaded SCF from:", cfg.gpw_file)
    print("PBC:", atoms.pbc)
    print("Cell:", atoms.cell, "\n")

    # --- 2. Build explicit 1D k-grid along z ---
    nkpts = cfg.npoints_kpath
    k_frac = np.linspace(0.0, 0.5, nkpts)   # fractional kz: 0 → Γ, 0.5 → Z
    kpts = np.zeros((nkpts, 3))
    kpts[:, 2] = k_frac

    print(f"Using {nkpts} k-points along z (fractional kz 0 → 0.5)")

    # --- 3. Non-selfconsistent band calculation with fixed density ---
    bs_calc = gs_calc.fixed_density(
        nbands=cfg.nbands_bands,
        kpts=kpts,
        symmetry='off',
        txt='zgnr_bands.txt'
    )

    # --- 4. Collect eigenvalues E[spin, k, band] ---
    nspins = bs_calc.wfs.nspins
    nbands = cfg.nbands_bands

    energies = np.zeros((nspins, nkpts, nbands), dtype=float)
    for s in range(nspins):
        for ik in range(nkpts):
            energies[s, ik, :] = bs_calc.get_eigenvalues(kpt=ik, spin=s)

    # Fermi level from ground-state calc
    efermi = gs_calc.get_fermi_level()
    E_rel = energies - efermi

    print(f"nspins = {nspins}, nkpts = {nkpts}, nbands = {nbands}")
    print(f"Fermi level (eV): {efermi:.6f}")

    # --- 5. k-axis for plotting: ka/π in [0,1] ---
    k_dimless = 2.0 * np.pi * k_frac     # ka in [0, π]
    k_plot = k_dimless / np.pi           # ka/π in [0, 1]
    X_ticks = [0.0, 1.0]
    X_labels = [r'$\Gamma$', r'$Z$']

    # --- 6. Plot all bands ---
    plt.figure(figsize=(5, 6))
    for s in range(nspins):
        for n in range(nbands):
            plt.plot(k_plot, E_rel[s, :, n], linewidth=0.4)
    plt.axhline(0.0, linestyle='--', linewidth=0.8, color='k')
    plt.xticks(X_ticks, X_labels)
    plt.xlabel(r'$k a / \pi$ (0 → $\Gamma$, 1 → Z)')
    plt.ylabel(r'$E - E_F$ (eV)')
    plt.title('ZGNR – all bands (DFT)')
    plt.tight_layout()
    plt.savefig(cfg.bands_full_png, dpi=300)
    plt.close()
    print(f"Saved full band plot to '{cfg.bands_full_png}'")

    # --- 7. Select π-like bands in energy window ---
    E_min_pi = cfg.E_min_pi
    E_max_pi = cfg.E_max_pi

    print(f"Selecting bands entering [{E_min_pi}, {E_max_pi}] eV around EF.")
    pi_band_indices_per_spin = []

    for s in range(nspins):
        mask = (E_rel[s] > E_min_pi) & (E_rel[s] < E_max_pi)
        band_indices = [n for n in range(nbands) if mask[:, n].any()]
        pi_band_indices_per_spin.append(band_indices)
        print(f"Spin {s}: π-like band indices = {band_indices}")

    # --- 8. Plot only π-like bands ---
    plt.figure(figsize=(5, 6))
    colors = ['gold', 'royalblue']
    spin_labels = ['spin up', 'spin down']

    for s in range(nspins):
        band_list = pi_band_indices_per_spin[s]
        for n in band_list:
            plt.plot(
                k_plot,
                E_rel[s, :, n],
                linewidth=1.0,
                color=colors[s % 2],
                label=spin_labels[s] if n == band_list[0] else None
            )

    plt.axhline(0.0, linestyle='--', linewidth=0.8, color='k')
    plt.xticks(X_ticks, X_labels)
    plt.ylim(E_min_pi, E_max_pi)
    plt.xlabel(r'$k a / \pi$ (0 → $\Gamma$, 1 → Z)')
    plt.ylabel(r'$E - E_F$ (eV)')
    plt.title('ZGNR – π-like bands (DFT)')
    if nspins > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.bands_pi_png, dpi=300)
    plt.close()
    print(f"Saved π-band plot to '{cfg.bands_pi_png}'")

    # --- 9. Save all band data for post-processing ---
    bands_up = np.array(pi_band_indices_per_spin[0], dtype=int)
    if nspins > 1:
        bands_dn = np.array(pi_band_indices_per_spin[1], dtype=int)
    else:
        bands_dn = np.array([], dtype=int)

    np.savez(
        cfg.bands_pi_npz,
        k_frac=k_frac,
        k_dimless=k_dimless,
        k_plot=k_plot,
        energies=energies,
        E_rel=E_rel,
        efermi=efermi,
        bands_up=bands_up,
        bands_dn=bands_dn
    )
    print(f"Saved band data to '{cfg.bands_pi_npz}'")


if __name__ == "__main__":
    main()
