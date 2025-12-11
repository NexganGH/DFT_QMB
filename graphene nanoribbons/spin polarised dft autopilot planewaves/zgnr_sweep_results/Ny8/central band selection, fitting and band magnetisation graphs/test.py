import numpy as np
import matplotlib.pyplot as plt

#### LINE 132 CHANGE NE THAME FO THE FILE CORRECTLY ADPTING IT TO THE NUMBER OF STRANDS

# ============================================================
# 1. Load all DFT bands from an existing npz
# ============================================================

def load_all_bands(npz_file, spin_index=0):
    data = np.load(npz_file)

    # --- k array ---
    if "k_dimless" in data.files:
        k = data["k_dimless"]
    elif "k" in data.files:
        k = data["k"]
    else:
        raise KeyError(f"No k array found in {npz_file}. "
                       f"Available keys: {data.files}")

    # --- energies: try various patterns ---
    E_all = None
    used_key = None

    for key in ["E_rel_all", "E_all", "E_rel", "E"]:
        if key in data.files:
            E_all = data[key]
            used_key = key
            break

    if E_all is None:
        if "E_rel_up" in data.files and "E_rel_dn" in data.files:
            E_all = np.stack([data["E_rel_up"], data["E_rel_dn"]], axis=0)
            used_key = "E_rel_up + E_rel_dn"
        elif "E_up" in data.files and "E_dn" in data.files:
            E_all = np.stack([data["E_up"], data["E_dn"]], axis=0)
            used_key = "E_up + E_dn"

    if E_all is None:
        raise KeyError(
            f"No suitable energy array found in {npz_file}. "
            f"Available keys: {data.files}"
        )

    print(f"Using energy key '{used_key}' from {npz_file}")

    if E_all.ndim == 3:
        E = E_all[spin_index]      # (Nk, nbands)
    elif E_all.ndim == 2:
        E = E_all
    else:
        raise ValueError(f"Energy array has unexpected shape: {E_all.shape}")

    return k, E


# ============================================================
# 2. Choose starting indices at Z (last k-point)
# ============================================================

def choose_start_indices_from_Z(E):
    """
    E : (Nk, nbands), energies relative to EF.

    Start from k = Z (last k-point), where the π bands are
    usually the cleanest, and choose valence / conduction there.
    """
    Nk, nbands = E.shape
    i0 = Nk - 1          # Z point
    Ek = E[i0]

    # valence: largest E < 0
    mask_val = Ek < 0.0
    if not np.any(mask_val):
        raise RuntimeError("No energies below EF at k=Z.")
    val_candidates = np.where(mask_val)[0]
    n_val = val_candidates[np.argmax(Ek[mask_val])]

    # conduction: smallest E > 0
    mask_cond = Ek > 0.0
    if not np.any(mask_cond):
        raise RuntimeError("No energies above EF at k=Z.")
    cond_candidates = np.where(mask_cond)[0]
    n_cond = cond_candidates[np.argmin(Ek[mask_cond])]

    print(f"Start from Z: i0 = {i0}")
    print(f"Valence index at Z:    {n_val}")
    print(f"Conduction index at Z: {n_cond}")

    return i0, n_val, n_cond


# ============================================================
# 3. Track a band by energy continuity
# ============================================================

def track_band_by_continuity(E, i0, n0):
    """
    Follow one band across k by choosing, at each neighbouring k,
    the level whose energy is closest to the previous energy.

    E  : (Nk, nbands)
    i0 : starting k index
    n0 : starting band index at k[i0]
    """
    Nk, nbands = E.shape
    idx = np.empty(Nk, dtype=int)
    idx[i0] = n0

    # backward in k (Z -> Γ)
    for i in range(i0 - 1, -1, -1):
        E_prev = E[i + 1, idx[i + 1]]
        diffs = np.abs(E[i] - E_prev)
        idx[i] = np.argmin(diffs)

    # forward in k just in case (though i0 is the last index)
    for i in range(i0 + 1, Nk):
        E_prev = E[i - 1, idx[i - 1]]
        diffs = np.abs(E[i] - E_prev)
        idx[i] = np.argmin(diffs)

    return idx


# ============================================================
# 4. Main: smooth extraction, plotting, saving
# ============================================================

def main():
    # ---- user settings ----
    npz_in = "zgnr_Ny8_M1_bands_pi.npz"   # your DFT bands file
    npz_out = "zgnr_pi_smooth_from_Z.npz"
    spin_index = 0                        # 0 or 1 if spin-polarised
    # -----------------------

    k, E = load_all_bands(npz_in, spin_index=spin_index)
    Nk, nbands = E.shape
    print(f"Loaded: Nk={Nk}, nbands={nbands}")

    # starting from Z
    i0, n_val0, n_cond0 = choose_start_indices_from_Z(E)

    # track smooth bands
    idx_val = track_band_by_continuity(E, i0, n_val0)
    idx_cond = track_band_by_continuity(E, i0, n_cond0)

    E_val = E[np.arange(Nk), idx_val]
    E_cond = E[np.arange(Nk), idx_cond]

    # ---- plot all bands + smooth π ----
    fig, ax = plt.subplots(figsize=(6, 5))

    for n in range(nbands):
        ax.plot(k/np.pi, E[:, n], lw=0.6, color="0.8", zorder=1)

    ax.plot(k/np.pi, E_val, lw=2.5, color="C1", label="π valence (smooth)")
    ax.plot(k/np.pi, E_cond, lw=2.5, color="C0", label="π conduction (smooth)")

    ax.axhline(0.0, ls="--", lw=0.8, color="k", alpha=0.6)
    ax.set_xlabel(r"$k a/\pi$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title("Smooth π Bands (tracking from Z)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("smooth_pi_from_Z.png", dpi=300)
    print("Saved plot as smooth_pi_from_Z.png")

    # ---- save for later post-processing ----
    np.savez(
        npz_out,
        k_dimless=k,
        E_val=E_val,
        E_cond=E_cond,
        idx_val=idx_val,
        idx_cond=idx_cond,
    )
    print(f"Saved smooth π bands (from Z) to: {npz_out}")


if __name__ == "__main__":
    main()
