import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------------------------------
# Make project root visible
# -------------------------------------------------
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.append(project_root)

# -------------------------------------------------
# Import MF solver + DOS
# -------------------------------------------------
from mft.magnetisation_mft import solve_zgnr_mf, compute_dos


# -------------------------------
# Folder helpers
# -------------------------------
def make_output_dirs(base_dir, N, U, t):
    run_dir = os.path.join(base_dir, f"N{N}", f"U{U:.4f}_t{t:.4f}")
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
    below = np.where(Ek <= mu)[0]
    above = np.where(Ek > mu)[0]
    ival = below[np.argmax(Ek[below])] if len(below) else None
    icond = above[np.argmin(Ek[above])] if len(above) else None
    return ival, icond


# -------------------------------
# Bands plot: 3-color scheme + in-plot N annotation
# -------------------------------
def plot_bands_three_colors(result, filename=None, datafile=None):
    k = result["k_grid"]
    E = result["E"]          # (2, Nk, dim)
    mu = float(result["mu"])
    Nk = len(k)
    dim = E.shape[-1]
    N = int(result["Ny"])    # your solver uses Ny internally, but we label it as N

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

        # If a band crosses mu, color by majority side
        frac_below = np.mean(y <= mu)
        if is_below:
            c = col_val
        elif is_above:
            c = col_con
        else:
            c = col_val if frac_below >= 0.5 else col_con

        plt.plot(k, y, lw=lw_other, alpha=alpha_other, color=c)

    # Highlight the two central bands
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
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)

    # In-plot annotation: N
    ax = plt.gca()
    ax.text(
        0.97, 0.97,
        f"$N = {N}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8),
    )

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
            N=N,
        )


# -------------------------------
# DOS plot + in-plot N annotation
# -------------------------------
def plot_dos_styled(Egrid, dos, mu=0.0, N=None, filename=None, datafile=None):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, dos, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb. units)")
    plt.grid(True, alpha=0.25)

    if N is not None:
        ax = plt.gca()
        ax.text(
            0.97, 0.97,
            f"$N = {int(N)}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8),
        )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(datafile, Egrid=Egrid, dos=dos, mu=float(mu), N=int(N) if N is not None else -1)


# -------------------------------
# Magnetization plot + marker-only legend + in-plot N annotation
# -------------------------------
def plot_magnetization_profile_styled(result, filename=None, datafile=None):
    mA = np.array(result["mA"], dtype=float)
    mB = np.array(result["mB"], dtype=float)
    N = int(result["Ny"])
    chains = np.arange(N)

    col_A = "#2E5EAA"
    col_B = "#2B8C7E"

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(chains, mA, marker="o", lw=1.8, color=col_A)
    plt.plot(chains, mB, marker="s", lw=1.8, color=col_B)

    plt.axhline(0.0, ls="--", lw=1.2, alpha=0.8)
    plt.xlabel("Zigzag strand index")
    plt.ylabel("Magnetization m")
    plt.grid(True, alpha=0.25)

    # Marker-only legend
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=col_A, markeredgecolor=col_A, label="A"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=col_B, markeredgecolor=col_B, label="B"),
    ]
    plt.legend(handles=handles, frameon=False, loc="upper right")

    # In-plot annotation: N (top-left to avoid legend)
    ax = plt.gca()
    ax.text(
        0.03, 0.97,
        f"$N = {N}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8),
    )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(datafile, chains=chains, mA=mA, mB=mB, N=N)


# ===========================================================
# Sweep N = 2..20 and collect images
# ===========================================================
if __name__ == "__main__":
    # Global parameters (same for all N)
    t = 2.4
    U = 2.7
    Nk = 400
    filling = 1.0

    base_dir = "postproc_outputs"

    # Common folders collecting ALL N images (for later GIF)
    all_dirs = {
        "bands": os.path.join(base_dir, "_ALL", "bands"),
        "dos": os.path.join(base_dir, "_ALL", "dos"),
        "mag": os.path.join(base_dir, "_ALL", "magnetization"),
    }
    for d in all_dirs.values():
        os.makedirs(d, exist_ok=True)

    for N in range(1, 21):
        print("\n" + "=" * 70)
        print(f"Running N = {N}  |  t = {t} eV  U = {U} eV  Nk = {Nk}  filling = {filling}")
        print("=" * 70)

        # Run solver (solver parameter name is Ny; we treat it as N in labels)
        result = solve_zgnr_mf(
            Ny=N, U=U, t=t, a=1.0, Nk=Nk,
            filling=filling, max_iter=300, mix=0.1, tol=1e-5, verbose=True
        )

        # Per-run folders
        dirs = make_output_dirs(base_dir, N, U, t)

        # --- Bands ---
        bands_png = os.path.join(dirs["bands"], "bands_three_colors.png")
        bands_npz = os.path.join(dirs["data"], "bands_three_colors.npz")
        plot_bands_three_colors(result, filename=bands_png, datafile=bands_npz)

        bands_png_all = os.path.join(all_dirs["bands"], f"bands_N{N:02d}_U{U:.4f}_t{t:.4f}.png")
        shutil.copy2(bands_png, bands_png_all)

        # --- DOS ---
        Egrid, dos = compute_dos(result, nE=600, eta=0.05)
        dos_png = os.path.join(dirs["dos"], "dos.png")
        dos_npz = os.path.join(dirs["data"], "dos.npz")
        plot_dos_styled(Egrid, dos, mu=result["mu"], N=N, filename=dos_png, datafile=dos_npz)

        dos_png_all = os.path.join(all_dirs["dos"], f"dos_N{N:02d}_U{U:.4f}_t{t:.4f}.png")
        shutil.copy2(dos_png, dos_png_all)

        # --- Magnetization ---
        mag_png = os.path.join(dirs["mag"], "mag_profile.png")
        mag_npz = os.path.join(dirs["data"], "mag_profile.npz")
        plot_magnetization_profile_styled(result, filename=mag_png, datafile=mag_npz)

        mag_png_all = os.path.join(all_dirs["mag"], f"mag_N{N:02d}_U{U:.4f}_t{t:.4f}.png")
        shutil.copy2(mag_png, mag_png_all)

        # --- Run summary ---
        summary_npz = os.path.join(dirs["data"], "run_summary.npz")
        np.savez(
            summary_npz,
            N=N, U=U, t=t, Nk=Nk, filling=filling,
            mu=float(result["mu"]),
            k_grid=result["k_grid"],
            E=result["E"],
            mA=result["mA"],
            mB=result["mB"],
            nA=result["nA"],
            nB=result["nB"],
        )

        print("Saved per-run folders in:", dirs["run"])
        print("Collected PNGs in:", os.path.join(base_dir, "_ALL"))

    print("\nDONE. All N=2..20 completed.")
    print("All images collected in:")
    print("  ", os.path.join(base_dir, "_ALL", "bands"))
    print("  ", os.path.join(base_dir, "_ALL", "dos"))
    print("  ", os.path.join(base_dir, "_ALL", "magnetization"))
