import numpy as np
import matplotlib.pyplot as plt

def main():
    # file produced by the smoothing script
    npz_file = "zgnr_pi_smooth_from_Z.npz"

    data = np.load(npz_file)
    k = data["k_dimless"]
    E_val = data["E_val"]
    E_cond = data["E_cond"]

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(k/np.pi, E_val, lw=2.5, color="C1", label="π valence (smooth)")
    ax.plot(k/np.pi, E_cond, lw=2.5, color="C0", label="π conduction (smooth)")

    ax.axhline(0.0, ls="--", lw=0.8, color="k", alpha=0.6)

    ax.set_xlabel(r"$k a / \pi$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title("Smooth π Bands (DFT → Smoothed)")
    ax.legend()

    fig.tight_layout()
    fig.savefig("smooth_pi_plot.png", dpi=300)

    print("Saved plot as smooth_pi_plot.png")

if __name__ == "__main__":
    main()
