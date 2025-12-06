# 05_zgnr_central_bands.py
#
# Extract the "central" π bands: the valence and conduction band
# closest to EF for each spin, from the data produced by 04_zgnr_bands_pi.py.

import numpy as np
import matplotlib.pyplot as plt
import config_zgnr as cfg


def find_central_bands_for_spin(E_rel, band_list):
    """
    Given E_rel[s, k, n] and a list of band indices 'band_list' that are π-like
    for a given spin channel, pick:
      - valence band: <E> < 0 and closest to 0
      - conduction band: <E> > 0 and closest to 0

    If one of those doesn't exist (all >0 or all <0), we fall back to the bands
    with smallest |<E>| overall.
    """
    band_list = np.array(band_list, dtype=int)
    if band_list.size == 0:
        return None, None, None, None

    # E_rel has shape (nkpts, nbands) for a fixed spin.
    # Compute average energy of each candidate band over k.
    E_mean = E_rel[:, band_list].mean(axis=0)

    # --- valence: negative mean energy closest to 0 ---
    mask_val = E_mean < 0.0
    if np.any(mask_val):
        idx = np.argmin(np.abs(E_mean[mask_val]))
        val_band = band_list[mask_val][idx]
    else:
        # fallback: band with mean energy closest to 0
        idx = np.argmin(np.abs(E_mean))
        val_band = band_list[idx]

    # --- conduction: positive mean energy closest to 0 ---
    mask_cond = E_mean > 0.0
    if np.any(mask_cond):
        idx = np.argmin(np.abs(E_mean[mask_cond]))
        cond_band = band_list[mask_cond][idx]
    else:
        # fallback: pick the band with next-smallest |<E>| if possible
        order = np.argsort(np.abs(E_mean))
        if band_list.size == 1:
            cond_band = band_list[order[0]]
        else:
            # ensure cond_band can be different from val_band
            cond_band = band_list[order[1]]

    # Extract the k-dependent dispersions
    E_val = E_rel[:, val_band]
    E_cond = E_rel[:, cond_band]

    return val_band, cond_band, E_val, E_cond


def main():
    # --- 1. Load π-band data saved by 04_zgnr_bands_pi.py ---
    data = np.load(cfg.bands_pi_npz, allow_pickle=True)

    # k-axis: we saved k_dimless = k*a in [0, π] and/or k_plot = k*a/π
    if "k_plot" in data:
        k_plot = data["k_plot"]           # dimensionless ka/π in [0, 1]
    else:
        # fallback: build from k_dimless
        k_plot = data["k_dimless"] / np.pi

    E_rel = data["E_rel"]                # shape (nspins, nkpts, nbands)
    efermi = float(data["efermi"])
    bands_up = data["bands_up"]          # indices of π bands for spin 0
    bands_dn = data["bands_dn"]          # indices of π bands for spin 1 (may be empty)

    nspins, nkpts, nbands = E_rel.shape
    print("Loaded band data from:", cfg.bands_pi_npz)
    print("nspins =", nspins, "nkpts =", nkpts, "nbands =", nbands)
    print("π band indices (spin up):", bands_up)
    if bands_dn.size > 0:
        print("π band indices (spin dn):", bands_dn)

    # --- 2. Find central bands for each spin channel ---
    central_info = {}

    # Spin 0 (up)
    val_up_idx, cond_up_idx, E_val_up, E_cond_up = find_central_bands_for_spin(
        E_rel[0], bands_up
    )
    central_info["val_up_idx"] = val_up_idx
    central_info["cond_up_idx"] = cond_up_idx
    central_info["E_val_up"] = E_val_up
    central_info["E_cond_up"] = E_cond_up

    print("\nSpin UP:")
    print("  central valence band index:", val_up_idx)
    print("  central conduction band index:", cond_up_idx)

    # Spin 1 (down), if present
    if nspins > 1 and bands_dn.size > 0:
        val_dn_idx, cond_dn_idx, E_val_dn, E_cond_dn = find_central_bands_for_spin(
            E_rel[1], bands_dn
        )
        central_info["val_dn_idx"] = val_dn_idx
        central_info["cond_dn_idx"] = cond_dn_idx
        central_info["E_val_dn"] = E_val_dn
        central_info["E_cond_dn"] = E_cond_dn

        print("\nSpin DOWN:")
        print("  central valence band index:", val_dn_idx)
        print("  central conduction band index:", cond_dn_idx)
    else:
        central_info["val_dn_idx"] = None
        central_info["cond_dn_idx"] = None
        central_info["E_val_dn"] = None
        central_info["E_cond_dn"] = None

    # --- 3. Plot central bands (spin-resolved) ---
    plt.figure(figsize=(6, 5))

    # Spin up bands
    if central_info["E_val_up"] is not None:
        plt.plot(
            k_plot,
            central_info["E_val_up"],
            "C0-",
            label="valence (↑)"
        )
    if central_info["E_cond_up"] is not None:
        plt.plot(
            k_plot,
            central_info["E_cond_up"],
            "C1-",
            label="conduction (↑)"
        )

    # Spin down bands (if exist)
    if central_info["E_val_dn"] is not None:
        plt.plot(
            k_plot,
            central_info["E_val_dn"],
            "C0--",
            label="valence (↓)"
        )
    if central_info["E_cond_dn"] is not None:
        plt.plot(
            k_plot,
            central_info["E_cond_dn"],
            "C1--",
            label="conduction (↓)"
        )

    plt.axhline(0.0, ls="--", lw=0.8, color="k")
    plt.xlabel(r"$k a / \pi$ (0 → $\Gamma$, 1 → Z)")
    plt.ylabel(r"$E - E_F$ (eV)")
    plt.title("Central π bands (closest to $E_F$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.central_bands_png, dpi=300)
    plt.close()

    print(f"\nSaved central bands plot to '{cfg.central_bands_png}'")

    # --- 4. Save data for TB comparison / post-processing ---
    np.savez(
        cfg.central_bands_npz,
        k_plot=k_plot,
        efermi=efermi,
        **central_info
    )

    print(f"Saved central band data to '{cfg.central_bands_npz}'")


if __name__ == "__main__":
    main()
