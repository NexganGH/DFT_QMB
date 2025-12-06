import numpy as np
import matplotlib.pyplot as plt

NPZ_FILE = "zgnr_pi_bands.npz"
SPIN_INDEX = 1   # 0 = spin up, 1 = spin down


def load_dft(npz_file=NPZ_FILE):
    data = np.load(npz_file)
    print("Keys in npz:", data.files)

    # --- k-axis ---
    if "k_dimless" in data.files:
        k_raw = data["k_dimless"]
        source = "k_dimless"
    elif "k" in data.files:
        k_raw = data["k"]
        source = "k"
    elif "k_dist" in data.files:
        k_raw = data["k_dist"]
        source = "k_dist"
    else:
        raise KeyError("No k-array found in npz.")

    print(f"Using '{source}' as raw k-axis.")
    print("Raw k range:", k_raw[0], "→", k_raw[-1])

    # convert to ka in [0, π]
    k_min = float(k_raw.min())
    k_max = float(k_raw.max())

    if (k_min >= -1e-6
        and np.isclose(k_max, np.pi, rtol=1e-2, atol=1e-2)):
        k_ka = k_raw
        print("Interpreting raw k as dimensionless ka.")
    else:
        k_ka = k_raw / k_max * np.pi
        print("Rescaled raw k to ka in [0, π]. factor =", np.pi / k_max)

    print("Final ka range:", k_ka[0], "→", k_ka[-1])

    E_rel_all = data["E_rel_all"]
    efermi = data["efermi"]

    # π-band indices for this spin, if present
    if SPIN_INDEX == 1 and "bands_dn" in data.files:
        pi_indices = data["bands_dn"]
    elif SPIN_INDEX == 0 and "bands_up" in data.files:
        pi_indices = data["bands_up"]
    else:
        pi_indices = None

    print(f"π-like band indices for spin {SPIN_INDEX}:", pi_indices)

    return k_ka, E_rel_all, efermi, pi_indices


def plot_bands_with_pi(k_ka, E_rel_all, pi_indices, spin_index=SPIN_INDEX,
                       outfile="zgnr_full_bands_pi_highlight.png"):
    E_spin = E_rel_all[spin_index]   # (Nk, nbands)
    Nk, nbands = E_spin.shape

    plt.figure(figsize=(5, 6))

    # 1. all bands in light grey
    for n in range(nbands):
        plt.plot(k_ka, E_spin[:, n],
                 color="0.8", lw=0.5, zorder=1)

    # 2. π-like bands in colour (if we have indices)
    if pi_indices is not None and len(pi_indices) > 0:
        colors = ["C0", "C1", "C2", "C3", "C4"]
        for i, b in enumerate(pi_indices):
            plt.plot(k_ka, E_spin[:, b],
                     color=colors[i % len(colors)],
                     lw=2.0,
                     label=f"π band index {b}")
    else:
        print("WARNING: no π-band indices found in npz for this spin.")

    plt.axhline(0.0, ls="--", color="k", alpha=0.6)
    plt.xlabel(r"$ka$")
    plt.ylabel(r"$E - E_F$ (eV)")
    plt.xticks([0.0, np.pi], [r"$0$", r"$\pi$"])
    plt.title(f"Full bands (spin {spin_index}), π-like bands highlighted")
    if pi_indices is not None and len(pi_indices) > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved plot to '{outfile}'")


if __name__ == "__main__":
    k_ka, E_rel_all, efermi, pi_indices = load_dft(NPZ_FILE)
    print("E_rel_all shape =", E_rel_all.shape)

    plot_bands_with_pi(k_ka, E_rel_all, pi_indices,
                       spin_index=SPIN_INDEX,
                       outfile="zgnr_full_bands_pi_highlight.png")

