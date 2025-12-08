import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # ============================
    # USER PARAMETER: Ny
    # ============================
    Ny = 1  # <-- change this if you reuse the script in Ny2, Ny4, ...

    # File produced by your "central two above from Z" script
    # For Ny=1 this is: zgnr_Ny1_M1_pi_two_above_from_Z.npz
    infile = f"zgnr_Ny{Ny}_M1_pi_two_above_from_Z.npz"

    if not os.path.exists(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")

    data = np.load(infile)
    k = data["k_dimless"]

    # Depending on how the extractor was written, the two bands may be
    # stored as E1/E2 or as E_val/E_cond. We handle both cases.
    if "E1" in data.files and "E2" in data.files:
        E_val = data["E1"]
        E_cond = data["E2"]
    elif "E_val" in data.files and "E_cond" in data.files:
        E_val = data["E_val"]
        E_cond = data["E_cond"]
    else:
        raise KeyError(
            f"Could not find central bands in {infile}. "
            f"Available keys: {data.files}"
        )

    print(f"Loaded central bands from {infile}")
    print(f"Nk = {len(k)}")

    # ============================================================
    # 1. Plot ONLY the two central bands
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(k / np.pi, E_val, lw=2.5, color="C1", label="π valence (central)")
    ax.plot(k / np.pi, E_cond, lw=2.5, color="C0", label="π conduction (central)")

    ax.axhline(0.0, ls="--", color="k", lw=0.8, alpha=0.6)

    ax.set_xlabel(r"$k a/\pi$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title(f"Central π bands, Ny={Ny}")
    ax.legend()

    fig.tight_layout()
    fig.savefig("smooth_pi_plot.png", dpi=300)
    print("Saved clean central-band plot as 'smooth_pi_plot.png'")

    # ============================================================
    # 2. Save in STANDARD name for later post-processing
    # ============================================================
    outfile = "zgnr_pi_smooth_from_Z.npz"

    np.savez(
        outfile,
        k_dimless=k,
        E_val=E_val,
        E_cond=E_cond,
    )

    print(f"Saved unified file for post-processing: '{outfile}'")


if __name__ == "__main__":
    main()
