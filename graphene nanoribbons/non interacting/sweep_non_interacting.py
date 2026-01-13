import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# 1) Non-interacting ZGNR tight-binding (your model)
# ===========================================================

def H_zgnr_k(k, N, t=-2.7, a=1.0):
    alpha = 2.0 * np.cos(k * a / 2.0)
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=complex)

    for n in range(1, N + 1):
        iA = 2 * (n - 1)
        iB = 2 * (n - 1) + 1

        # A_n <-> B_n
        H[iA, iB] += t * alpha
        H[iB, iA] += t * alpha

        # A_n <-> B_{n-1}
        if n > 1:
            iB_prev = 2 * (n - 2) + 1
            H[iA, iB_prev] += t
            H[iB_prev, iA] += t

        # B_n <-> A_{n+1}
        if n < N:
            iA_next = 2 * n
            H[iB, iA_next] += t
            H[iA_next, iB] += t

    return H


def bands_zgnr(N, nk=400, t=-2.7, a=1.0):
    ks = np.linspace(-np.pi / a, np.pi / a, nk, endpoint=False)
    nb = 2 * N
    E = np.zeros((nk, nb), dtype=float)

    for i, k in enumerate(ks):
        w, _ = np.linalg.eigh(H_zgnr_k(k, N, t=t, a=a))
        E[i, :] = np.sort(w.real)

    return ks, E


def dos_from_bands(energies, n_E=1000, eta=0.05):
    eps = energies.ravel()
    Emin, Emax = eps.min(), eps.max()

    pad = 0.2 * (Emax - Emin)
    Emin -= pad
    Emax += pad

    E_grid = np.linspace(Emin, Emax, n_E)

    nk = energies.shape[0]
    x = E_grid[:, None] - eps[None, :]
    gauss = np.exp(-0.5 * (x / eta) ** 2) / (np.sqrt(2 * np.pi) * eta)
    DOS = gauss.sum(axis=1) / nk
    return E_grid, DOS


# ===========================================================
# 2) Central bands (fixed indices -> avoids vertical artifacts)
# ===========================================================

def extract_central_bands_fixed(E, N):
    """
    For sorted eigenvalues (nbands=2N):
      central valence   = index N-1
      central conduction = index N
    """
    val = E[:, N - 1]
    con = E[:, N]
    return val, con


# ===========================================================
# 3) Plot helpers (same style conventions)
# ===========================================================

def add_corner_label(ax, text, where="tr"):
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


def plot_bands_three_colors(ks, E, N, mu=0.0, filename=None, datafile=None):
    # Harmonized palette
    col_val = "#2E5EAA"   # deep blue
    col_con = "#D89C2B"   # warm amber
    col_ctr = "#C23B3B"   # muted red

    lw_other = 0.9
    lw_central = 2.2
    alpha_other = 0.85

    val_curve, con_curve = extract_central_bands_fixed(E, N)

    plt.figure(figsize=(7.2, 4.6))

    nb = E.shape[1]
    for b in range(nb):
        y = E[:, b]
        is_below = np.all(y <= mu)
        is_above = np.all(y > mu)
        frac_below = np.mean(y <= mu)

        if is_below:
            c = col_val
        elif is_above:
            c = col_con
        else:
            c = col_val if frac_below >= 0.5 else col_con

        plt.plot(ks, y, lw=lw_other, alpha=alpha_other, color=c)

    # Highlight central
    plt.plot(ks, val_curve, lw=lw_central, color=col_ctr)
    plt.plot(ks, con_curve, lw=lw_central, color=col_ctr)

    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    add_corner_label(ax, f"$N = {N}$", where="tr")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(datafile, k_grid=ks, E=E, mu=float(mu), N=int(N))


def plot_central_bands_only(ks, val_curve, con_curve, N, mu=0.0, filename=None, datafile=None):
    col_ctr = "#C23B3B"
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(ks, val_curve, lw=2.2, color=col_ctr)
    plt.plot(ks, con_curve, lw=2.2, color=col_ctr)

    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    add_corner_label(ax, f"$N = {N}$", where="tr")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(
            datafile,
            k_grid=ks,
            mu=float(mu),
            valence_central=val_curve,
            conduction_central=con_curve,
            N=int(N),
        )


def plot_dos_styled(Egrid, DOS, N, mu=0.0, filename=None, datafile=None):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, DOS, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb. units)")
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    add_corner_label(ax, f"$N = {N}$", where="tr")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile is not None:
        np.savez(datafile, Egrid=Egrid, dos=DOS, mu=float(mu), N=int(N))


# ===========================================================
# 4) Folder helpers + sweep
# ===========================================================

def make_output_dirs(base_dir, N, t):
    run_dir = os.path.join(base_dir, f"N{N}", f"t{t:.4f}")
    dirs = {
        "run": run_dir,
        "bands": os.path.join(run_dir, "bands"),
        "dos": os.path.join(run_dir, "dos"),
        "central": os.path.join(run_dir, "central_bands"),
        "data": os.path.join(run_dir, "data"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


if __name__ == "__main__":
    # ----------------- YOU CHOOSE -----------------
    N_min = 1
    N_max = 50

    t = -2.6          # eV
    a = 1.0
    nk = 1000
    eta = 0.05
    n_E = 1000
    mu = 0.0          # Fermi energy reference
    base_dir = "postproc_outputs_noninteracting"
    # ------------------------------------------------

    # Global collection folders (GIF-ready)
    all_dirs = {
        "bands": os.path.join(base_dir, "_ALL", "bands"),
        "dos": os.path.join(base_dir, "_ALL", "dos"),
        "central": os.path.join(base_dir, "_ALL", "central_bands"),
    }
    for d in all_dirs.values():
        os.makedirs(d, exist_ok=True)

    for N in range(N_min, N_max + 1):
        print(f"\nRunning non-interacting: N = {N}")

        ks, E = bands_zgnr(N=N, nk=nk, t=t, a=a)
        Egrid, DOS = dos_from_bands(E, n_E=n_E, eta=eta)
        val_curve, con_curve = extract_central_bands_fixed(E, N)

        dirs = make_output_dirs(base_dir, N, t)

        # Save full raw summary (everything)
        run_summary = os.path.join(dirs["data"], "run_summary.npz")
        np.savez(
            run_summary,
            N=int(N), t=float(t), a=float(a),
            nk=int(nk), eta=float(eta), n_E=int(n_E), mu=float(mu),
            k_grid=ks, E=E,
            Egrid=Egrid, dos=DOS,
            valence_central=val_curve,
            conduction_central=con_curve,
        )

        # --- Full bands plot (3 colors + central highlighted)
        bands_png = os.path.join(dirs["bands"], "bands_three_colors.png")
        bands_npz = os.path.join(dirs["data"], "bands_data.npz")
        plot_bands_three_colors(ks, E, N=N, mu=mu, filename=bands_png, datafile=bands_npz)

        # --- Central-only plot
        central_png = os.path.join(dirs["central"], "central_bands.png")
        central_npz = os.path.join(dirs["data"], "central_bands.npz")
        plot_central_bands_only(ks, val_curve, con_curve, N=N, mu=mu, filename=central_png, datafile=central_npz)

        # --- DOS plot
        dos_png = os.path.join(dirs["dos"], "dos.png")
        dos_npz = os.path.join(dirs["data"], "dos_data.npz")
        plot_dos_styled(Egrid, DOS, N=N, mu=mu, filename=dos_png, datafile=dos_npz)

        # Copy to _ALL (GIF-ready)
        shutil.copy2(bands_png, os.path.join(all_dirs["bands"], f"bands_N{N:02d}.png"))
        shutil.copy2(dos_png, os.path.join(all_dirs["dos"], f"dos_N{N:02d}.png"))
        shutil.copy2(central_png, os.path.join(all_dirs["central"], f"central_N{N:02d}.png"))

        print("Saved:", dirs["run"])

    print("\nDONE.")
    print("Collected PNGs (GIF-ready) in:")
    print("  ", all_dirs["bands"])
    print("  ", all_dirs["dos"])
    print("  ", all_dirs["central"])
