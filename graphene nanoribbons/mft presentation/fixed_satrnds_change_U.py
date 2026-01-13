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


# ===============================
# Helpers
# ===============================
def make_output_dirs(base_dir, N, t, U):
    run_dir = os.path.join(base_dir, f"N{N}", f"t{t:.4f}", f"U{U:.4f}")
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


def central_band_indices(Ek, mu):
    below = np.where(Ek <= mu)[0]
    above = np.where(Ek > mu)[0]
    ival = below[np.argmax(Ek[below])] if len(below) else None
    icond = above[np.argmin(Ek[above])] if len(above) else None
    return ival, icond


def add_corner_label(ax, text, where="tr"):
    # where: "tr" top-right, "tl" top-left
    if where == "tr":
        x, ha = 0.97, "right"
    else:
        x, ha = 0.03, "left"

    ax.text(
        x, 0.97, text,
        transform=ax.transAxes,
        ha=ha, va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8),
    )


# ===============================
# Plots
# ===============================
def plot_bands_three_colors(result, N, t, U, filename=None, datafile=None):
    k = result["k_grid"]
    E = result["E"]          # (2, Nk, 2N)
    mu = float(result["mu"])
    Nk = len(k)
    dim = E.shape[-1]

    Eavg = 0.5 * (E[0] + E[1])

    col_val = "#2E5EAA"
    col_con = "#D89C2B"
    col_ctr = "#C23B3B"

    lw_other = 0.9
    lw_central = 2.2
    alpha_other = 0.85

    plt.figure(figsize=(7.2, 4.6))

    for b in range(dim):
        y = Eavg[:, b]
        is_below = np.all(y <= mu)
        is_above = np.all(y > mu)
        frac_below = np.mean(y <= mu)

        if is_below:
            c = col_val
        elif is_above:
            c = col_con
        else:
            c = col_val if frac_below >= 0.5 else col_con

        plt.plot(k, y, lw=lw_other, alpha=alpha_other, color=c)

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

    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    add_corner_label(ax, f"$N={N}$\n$U/t={U/t:.2f}$", where="tr")

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
            N=int(N),
            t=float(t),
            U=float(U),
            U_over_t=float(U / t),
        )


def plot_dos_styled(Egrid, dos, mu, N, t, U, filename=None, datafile=None):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, dos, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb. units)")
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    add_corner_label(ax, f"$N={N}$\n$U/t={U/t:.2f}$", where="tr")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(
            datafile,
            Egrid=Egrid,
            dos=dos,
            mu=float(mu),
            N=int(N),
            t=float(t),
            U=float(U),
            U_over_t=float(U / t),
        )


def plot_magnetization_profile_styled(result, N, t, U, filename=None, datafile=None):
    mA = np.array(result["mA"], dtype=float)
    mB = np.array(result["mB"], dtype=float)
    chains = np.arange(N)

    col_A = "#2E5EAA"
    col_B = "#2B8C7E"

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(chains, mA, marker="o", lw=1.8, color=col_A)
    plt.plot(chains, mB, marker="s", lw=1.8, color=col_B)

    plt.axhline(0.0, ls="--", lw=1.2, alpha=0.8)
    plt.xlabel("Zigzag chain (across width)")
    plt.ylabel("Magnetization m")
    plt.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=col_A, markeredgecolor=col_A, label="A"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=col_B, markeredgecolor=col_B, label="B"),
    ]
    plt.legend(handles=handles, frameon=False, loc="upper right")

    ax = plt.gca()
    add_corner_label(ax, f"$N={N}$\n$U/t={U/t:.2f}$", where="tl")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(
            datafile,
            chains=chains,
            mA=mA,
            mB=mB,
            N=int(N),
            t=float(t),
            U=float(U),
            U_over_t=float(U / t),
        )


# ===============================
# Main: sweep U
# ===============================
if __name__ == "__main__":
    # -------- YOU CHOOSE THESE --------
    N = 6                 # fixed number of strands
    t = 2.4                # eV (fixed)
    dU_over_t = 0.10       # step in units of t (you choose)
    Umin_over_t = 1.0
    Umax_over_t = 3.0

    Nk = 400
    filling = 1.0
    max_iter = 300
    mix = 0.1
    tol = 1e-5
    eta = 0.05
    # ---------------------------------

    base_dir = "postproc_outputs_U_sweep"

    # GIF-ready collection folders (all U)
    all_dirs = {
        "bands": os.path.join(base_dir, "_ALL", "bands"),
        "dos": os.path.join(base_dir, "_ALL", "dos"),
        "mag": os.path.join(base_dir, "_ALL", "magnetization"),
    }
    for d in all_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Build U list in eV (multiples of t)
    U_over_t_list = np.arange(Umin_over_t, Umax_over_t + 1e-12, dU_over_t)
    U_list = t * U_over_t_list

    for U in U_list:
        U_over_t = U / t
        print("\n" + "=" * 70)
        print(f"Running N={N} | t={t:.4f} eV | U={U:.4f} eV (U/t={U_over_t:.2f})")
        print("=" * 70)

        # Run solver
        result = solve_zgnr_mf(
            Ny=N, U=U, t=t, a=1.0, Nk=Nk,
            filling=filling, max_iter=max_iter, mix=mix, tol=tol, verbose=True
        )

        # Per-run folders
        dirs = make_output_dirs(base_dir, N, t, U)

        # --- Bands ---
        bands_png = os.path.join(dirs["bands"], "bands_three_colors.png")
        bands_npz = os.path.join(dirs["data"], "bands_three_colors.npz")
        plot_bands_three_colors(result, N=N, t=t, U=U, filename=bands_png, datafile=bands_npz)

        bands_png_all = os.path.join(all_dirs["bands"], f"bands_N{N:02d}_Ut{U_over_t:05.2f}.png")
        shutil.copy2(bands_png, bands_png_all)

        # --- DOS ---
        Egrid, dos = compute_dos(result, nE=600, eta=eta)
        dos_png = os.path.join(dirs["dos"], "dos.png")
        dos_npz = os.path.join(dirs["data"], "dos.npz")
        plot_dos_styled(Egrid, dos, mu=result["mu"], N=N, t=t, U=U, filename=dos_png, datafile=dos_npz)

        dos_png_all = os.path.join(all_dirs["dos"], f"dos_N{N:02d}_Ut{U_over_t:05.2f}.png")
        shutil.copy2(dos_png, dos_png_all)

        # --- Magnetization ---
        mag_png = os.path.join(dirs["mag"], "mag_profile.png")
        mag_npz = os.path.join(dirs["data"], "mag_profile.npz")
        plot_magnetization_profile_styled(result, N=N, t=t, U=U, filename=mag_png, datafile=mag_npz)

        mag_png_all = os.path.join(all_dirs["mag"], f"mag_N{N:02d}_Ut{U_over_t:05.2f}.png")
        shutil.copy2(mag_png, mag_png_all)

        # --- Full raw summary ---
        summary_npz = os.path.join(dirs["data"], "run_summary.npz")
        np.savez(
            summary_npz,
            N=int(N), t=float(t), U=float(U), U_over_t=float(U_over_t),
            Nk=int(Nk), filling=float(filling),
            mu=float(result["mu"]),
            k_grid=result["k_grid"],
            E=result["E"],     # full spin-resolved eigenvalues
            mA=result["mA"],
            mB=result["mB"],
            nA=result["nA"],
            nB=result["nB"],
        )

        print("Saved per-run folders in:", dirs["run"])

    print("\nDONE.")
    print("Collected images (GIF-ready) in:")
    print("  ", all_dirs["bands"])
    print("  ", all_dirs["dos"])
    print("  ", all_dirs["mag"])
