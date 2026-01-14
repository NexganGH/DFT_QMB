import os
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Make project root visible
# -------------------------------------------------
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.append(project_root)

# -------------------------------------------------
# Import MF solver
# -------------------------------------------------
from mft.magnetisation_mft import solve_zgnr_mf, compute_dos

# -------------------------------
# Folder helpers
# -------------------------------
def make_output_dirs(base_dir, Ny, U, t):
    run_dir = os.path.join(base_dir, f"Ny{Ny}", f"U{U:.4f}_t{t:.4f}")
    dirs = {
        "run": run_dir,
        "bands": os.path.join(run_dir, "bands"),
        "dos": os.path.join(run_dir, "dos"),
        "mag": os.path.join(run_dir, "magnetization"),
        "data": os.path.join(run_dir, "data"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# -------------------------------
# Central bands selection (spin-averaged)
# -------------------------------
def central_band_indices(Ek, mu):
    """
    Ek: (dim,) energies at fixed k (spin-averaged).
    Return (ival, icond) indices of bands closest to mu from below/above.
    """
    below = np.where(Ek <= mu)[0]
    above = np.where(Ek > mu)[0]
    if len(below) == 0:
        ival = None
    else:
        ival = below[np.argmax(Ek[below])]
    if len(above) == 0:
        icond = None
    else:
        icond = above[np.argmin(Ek[above])]
    return ival, icond


# -------------------------------
# Bands plot: 3-color scheme
# -------------------------------
def plot_bands_three_colors(result, filename=None, datafile=None):
    k = result["k_grid"]
    E = result["E"]          # (2, Nk, dim)
    mu = float(result["mu"])
    Nk = len(k)
    dim = E.shape[-1]

    # Spin-averaged energies -> (Nk, dim)
    Eavg = 0.5 * (E[0] + E[1])

    # Harmonized palette (3 colors)
    col_val = "#2E5EAA"   # deep blue
    col_con = "#D89C2B"   # warm amber
    col_ctr = "#C23B3B"   # muted red

    lw_other = 0.9
    lw_central = 2.2
    alpha_other = 0.85

    plt.figure(figsize=(7.2, 4.6))

    # Plot all bands with consistent coloring below/above mu
    for b in range(dim):
        y = Eavg[:, b]
        is_below = np.all(y <= mu)
        is_above = np.all(y > mu)

        # If a band crosses mu, we still color it by the majority side
        frac_below = np.mean(y <= mu)
        if is_below:
            c = col_val
        elif is_above:
            c = col_con
        else:
            c = col_val if frac_below >= 0.5 else col_con

        plt.plot(k, y, lw=lw_other, alpha=alpha_other, color=c)

    # Highlight the two central bands (closest to mu at each k)
    # We collect their "band indices" varying with k, so we plot as points/segments.
    val_curve = np.full(Nk, np.nan)
    con_curve = np.full(Nk, np.nan)
    for ik in range(Nk):
        ival, icond = central_band_indices(Eavg[ik], mu)
        if ival is not None:
            val_curve[ik] = Eavg[ik, ival]
        if icond is not None:
            con_curve[ik] = Eavg[ik, icond]

    plt.plot(k, val_curve, lw=lw_central, color=col_ctr)
    plt.plot(k, con_curve, lw=lw_central, color=col_ctr)

    # Fermi level line
    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)

    plt.xlabel("k")
    plt.ylabel("Energy (eV)")  # see note below about units
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(
            datafile,
            k_grid=k,
            Eavg=Eavg,
            mu=mu,
            valence_central=val_curve,
            conduction_central=con_curve,
        )


# -------------------------------
# DOS (keep your compute_dos, but save data + plot styling)
# -------------------------------
def plot_dos_styled(Egrid, dos, mu=0.0, filename=None, datafile=None):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, dos, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb. units)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(datafile, Egrid=Egrid, dos=dos, mu=float(mu))


# -------------------------------
# Magnetization profile styling + save data
# -------------------------------
from matplotlib.lines import Line2D

def plot_magnetization_profile_styled(result, filename=None, datafile=None):
    mA = np.array(result["mA"], dtype=float)
    mB = np.array(result["mB"], dtype=float)
    Ny = int(result["Ny"])
    chains = np.arange(Ny)

    col_A = "#2E5EAA"
    col_B = "#2B8C7E"

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(chains, mA, marker="o", lw=1.8, color=col_A)
    plt.plot(chains, mB, marker="s", lw=1.8, color=col_B)

    plt.axhline(0.0, ls="--", lw=1.2, alpha=0.8)
    plt.xlabel("Zigzag strand index")
    plt.ylabel("Magnetization m")
    plt.grid(True, alpha=0.25)

    # --- Legend: marker-only (no lines) ---
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=col_A, markeredgecolor=col_A, label="A"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=col_B, markeredgecolor=col_B, label="B"),
    ]
    plt.legend(handles=handles, frameon=False)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(datafile, chains=chains, mA=mA, mB=mB)



# ===========================================================
# Example usage in __main__
# ===========================================================
if __name__ == "__main__":
    # --- your parameters ---
    Ny = 5
    t = 2.4
    U = 2.7
    Nk = 400
    filling = 1.0

    # Run solver (your existing function)
    result = solve_zgnr_mf(
        Ny=Ny, U=U, t=t, a=1.0, Nk=Nk,
        filling=filling, max_iter=300, mix=0.1, tol=1e-5, verbose=True
    )

    # Output folders
    base_dir = "postproc_outputs"
    dirs = make_output_dirs(base_dir, Ny, U, t)

    # --- Bands (3 colors + spin-averaged) ---
    bands_png = os.path.join(dirs["bands"], "bands_three_colors.png")
    bands_npz = os.path.join(dirs["data"], "bands_three_colors.npz")
    plot_bands_three_colors(result, filename=bands_png, datafile=bands_npz)
    print("Saved:", bands_png)
    print("Saved:", bands_npz)

    # --- DOS ---
    Egrid, dos = compute_dos(result, nE=600, eta=0.05)
    dos_png = os.path.join(dirs["dos"], "dos.png")
    dos_npz = os.path.join(dirs["data"], "dos.npz")
    plot_dos_styled(Egrid, dos, mu=result["mu"], filename=dos_png, datafile=dos_npz)
    print("Saved:", dos_png)
    print("Saved:", dos_npz)

    # --- Magnetization ---
    mag_png = os.path.join(dirs["mag"], "mag_profile.png")
    mag_npz = os.path.join(dirs["data"], "mag_profile.npz")
    plot_magnetization_profile_styled(result, filename=mag_png, datafile=mag_npz)
    print("Saved:", mag_png)
    print("Saved:", mag_npz)

    # Also save a compact "run summary" for postprocessing
    summary_npz = os.path.join(dirs["data"], "run_summary.npz")
    np.savez(
        summary_npz,
        Ny=Ny, U=U, t=t, Nk=Nk, filling=filling,
        mu=float(result["mu"]),
        k_grid=result["k_grid"],
        E=result["E"],     # full spin-resolved eigenvalues if you want later
        mA=result["mA"],
        mB=result["mB"],
        nA=result["nA"],
        nB=result["nB"],
    )
    print("Saved:", summary_npz)
