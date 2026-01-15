# src/zgnr_mf.py
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ===========================================================
# 0) Utilities
# ===========================================================

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


def make_output_dirs(base_dir, mode_tag, N, t, U):
    """
    One consistent structure for MF outputs.

    base_dir/
      <mode_tag>/                (e.g. "sweep_N" or "sweep_U" or "single")
        N{N}/
          t{t:.4f}/
            U{U:.4f}/
              bands/
              dos/
              magnetization/
              data/
    """
    run_dir = os.path.join(base_dir, mode_tag, f"N{N}", f"t{t:.4f}", f"U{U:.4f}")
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


def make_all_collection_dirs(base_dir, mode_tag):
    """
    GIF-ready collection dirs:
      base_dir/<mode_tag>/_ALL/bands, dos, magnetization
    """
    all_dirs = {
        "bands": os.path.join(base_dir, mode_tag, "_ALL", "bands"),
        "dos": os.path.join(base_dir, mode_tag, "_ALL", "dos"),
        "mag": os.path.join(base_dir, mode_tag, "_ALL", "magnetization"),
    }
    for d in all_dirs.values():
        os.makedirs(d, exist_ok=True)
    return all_dirs


# ===========================================================
# 1) Core MF solver
# ===========================================================

def build_h0_zigzag(Ny, k, t=1.0, a=1.0):
    """
    2*Ny x 2*Ny tight-binding Hamiltonian for a zigzag nanoribbon.
    Basis: [A1, B1, A2, B2, ..., A_Ny, B_Ny].
    """
    dim = 2 * Ny
    H0 = np.zeros((dim, dim), dtype=np.complex128)

    for l in range(dim - 1):
        if l % 2 == 0:
            val = -2.0 * t * np.cos(k * a / 2.0)  # A_m -> B_m
        else:
            val = -t                              # B_m -> A_{m+1}

        H0[l, l + 1] = val
        H0[l + 1, l] = np.conjugate(val)

    return H0


def build_V_sigma(Ny, sigma, U, mA, mB):
    """
    Diagonal MF Hubbard potential:
      V = -U * sigma * m_{site}
    sigma = +1 (up) or -1 (down).
    """
    dim = 2 * Ny
    V = np.zeros((dim, dim), dtype=np.complex128)

    for l in range(dim):
        m_index = l // 2
        if l % 2 == 0:   # A site
            V[l, l] = -U * sigma * mA[m_index]
        else:            # B site
            V[l, l] = -U * sigma * mB[m_index]

    return V


def solve_zgnr_mf(
    Ny,
    U,
    t=1.0,
    a=1.0,
    Nk=200,
    filling=1.0,
    max_iter=200,
    mix=0.1,
    tol=1e-5,
    verbose=False,
    m0=0.05,
):
    """
    Self-consistent mean-field Hubbard solution for zigzag nanoribbon.

    Returns dict with:
      k_grid, E (2,Nk,2Ny), Uvec, mA,mB, nA,nB, mu, params...
    """
    dim = 2 * Ny

    # k-grid along 1D BZ
    k_grid = np.linspace(-np.pi / a, np.pi / a, Nk, endpoint=False)

    # electrons per cell
    nelec_per_cell = filling * dim
    nelec_total = int(round(nelec_per_cell * Nk))

    # Initial AFM edge guess
    mA = np.zeros(Ny)
    mB = np.zeros(Ny)
    mA[0] = +m0; mB[0] = +m0
    mA[-1] = -m0; mB[-1] = -m0

    # Storage
    E = np.zeros((2, Nk, dim), dtype=np.float64)              # [spin, k, band]
    Uvec = np.zeros((2, Nk, dim, dim), dtype=np.complex128)
    mu = 0.0

    # Iterate
    for it in range(max_iter):
        # 1) diagonalize
        for s_idx, sigma in enumerate([+1, -1]):
            for ik, k in enumerate(k_grid):
                H0 = build_h0_zigzag(Ny, k, t=t, a=a)
                V = build_V_sigma(Ny, sigma, U, mA, mB)
                H = H0 + V
                vals, vecs = np.linalg.eigh(H)
                E[s_idx, ik, :] = vals
                Uvec[s_idx, ik, :, :] = vecs

        # 2) find mu by filling the lowest nelec_total states
        E_flat = E.reshape(-1)
        idx_sorted = np.argsort(E_flat)

        total_states = len(E_flat)
        if nelec_total >= total_states:
            mu = E_flat[idx_sorted[-1]] + 1e-6
        else:
            eN_1 = E_flat[idx_sorted[nelec_total - 1]]
            eN = E_flat[idx_sorted[nelec_total]]
            mu = 0.5 * (eN_1 + eN)

        # 3) compute densities
        nA_up = np.zeros(Ny)
        nA_dn = np.zeros(Ny)
        nB_up = np.zeros(Ny)
        nB_dn = np.zeros(Ny)

        w_k = 1.0 / Nk

        for s_idx, sigma in enumerate([+1, -1]):
            for ik in range(Nk):
                vals = E[s_idx, ik, :]
                vecs = Uvec[s_idx, ik, :, :]
                occ_inds = np.where(vals <= mu)[0]

                for band in occ_inds:
                    v = vecs[:, band]
                    for m in range(Ny):
                        idxA = 2 * m
                        idxB = 2 * m + 1
                        probA = np.abs(v[idxA]) ** 2
                        probB = np.abs(v[idxB]) ** 2

                        if sigma == +1:
                            nA_up[m] += w_k * probA
                            nB_up[m] += w_k * probB
                        else:
                            nA_dn[m] += w_k * probA
                            nB_dn[m] += w_k * probB

        nA_new = nA_up + nA_dn
        nB_new = nB_up + nB_dn
        mA_new = 0.5 * (nA_up - nA_dn)
        mB_new = 0.5 * (nB_up - nB_dn)

        # 4) mixing
        mA_mixed = (1.0 - mix) * mA + mix * mA_new
        mB_mixed = (1.0 - mix) * mB + mix * mB_new

        delta_m = max(np.max(np.abs(mA_mixed - mA)), np.max(np.abs(mB_mixed - mB)))
        mA, mB = mA_mixed, mB_mixed

        if verbose:
            print(f"Iter {it+1:3d}: mu={mu:+.6f}  max|Δm|={delta_m:.3e}  mA_edge=({mA[0]:+.4f},{mA[-1]:+.4f})")

        if delta_m < tol:
            break

    return {
        "k_grid": k_grid,
        "E": E,
        "Uvec": Uvec,
        "mA": mA,
        "mB": mB,
        "nA": nA_new,
        "nB": nB_new,
        "mu": float(mu),
        "t": float(t),
        "U": float(U),
        "Ny": int(Ny),
        "Nk": int(Nk),
        "a": float(a),
        "filling": float(filling),
        "mix": float(mix),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "m0": float(m0),
    }


def compute_dos(result, nE=600, E_min=None, E_max=None, eta=0.05):
    """
    DOS using Gaussian broadening from eigenvalues (spin resolved included).
    """
    Evals = result["E"]  # (2,Nk,dim)
    Nk = len(result["k_grid"])
    E_flat = Evals.reshape(-1)

    if E_min is None:
        E_min = E_flat.min() - 3 * eta
    if E_max is None:
        E_max = E_flat.max() + 3 * eta

    Egrid = np.linspace(E_min, E_max, nE)
    dos = np.zeros_like(Egrid)

    w_k = 1.0 / Nk
    pref = 1.0 / (np.sqrt(2 * np.pi) * eta)

    for E0 in E_flat:
        dos += w_k * pref * np.exp(-0.5 * ((Egrid - E0) / eta) ** 2)

    return Egrid, dos


# ===========================================================
# 2) Plotting
# ===========================================================

def central_band_indices(Ek, mu):
    below = np.where(Ek <= mu)[0]
    above = np.where(Ek > mu)[0]
    ival = below[np.argmax(Ek[below])] if len(below) else None
    icond = above[np.argmin(Ek[above])] if len(above) else None
    return ival, icond


def plot_bands_three_colors(result, filename=None, datafile=None, annotate_text=None):
    k = result["k_grid"]
    E = result["E"]          # (2,Nk,dim)
    mu = float(result["mu"])
    Nk = len(k)
    dim = E.shape[-1]

    # Spin-averaged bands for clean plots
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

    if annotate_text:
        ax = plt.gca()
        add_corner_label(ax, annotate_text, where="tr")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile:
        np.savez(
            datafile,
            k_grid=k,
            Eavg=Eavg,
            mu=mu,
            valence_central=val_curve,
            conduction_central=con_curve,
        )


def plot_dos_styled(Egrid, dos, mu=0.0, filename=None, datafile=None, annotate_text=None):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, dos, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb. units)")
    plt.grid(True, alpha=0.25)

    if annotate_text:
        ax = plt.gca()
        add_corner_label(ax, annotate_text, where="tr")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile:
        np.savez(datafile, Egrid=Egrid, dos=dos, mu=float(mu))


def plot_magnetization_profile_styled(result, filename=None, datafile=None, annotate_text=None):
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

    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=col_A, markeredgecolor=col_A, label="A"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=col_B, markeredgecolor=col_B, label="B"),
    ]
    plt.legend(handles=handles, frameon=False, loc="upper right")

    if annotate_text:
        ax = plt.gca()
        add_corner_label(ax, annotate_text, where="tl")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()

    if datafile:
        np.savez(datafile, chains=chains, mA=mA, mB=mB, N=N)


# ===========================================================
# 3) One-stop routines: single, sweep N, sweep U (fixed N)
# ===========================================================

def run_single_case(
    *,
    N,
    t,
    U,
    a,
    Nk,
    filling,
    max_iter,
    mix,
    tol,
    m0,
    dos_nE,
    dos_eta,
    base_dir,
    mode_tag="single",
    copy_to_all=True,
    verbose=False,
):
    """
    Runs MF for one (N,U), saves:
      - run_summary.npz (full info)
      - plots + plot data npz
      - optionally copies images to _ALL
    """
    result = solve_zgnr_mf(
        Ny=N, U=U, t=t, a=a, Nk=Nk,
        filling=filling, max_iter=max_iter, mix=mix, tol=tol,
        verbose=verbose, m0=m0,
    )

    dirs = make_output_dirs(base_dir, mode_tag, N, t, U)
    all_dirs = make_all_collection_dirs(base_dir, mode_tag) if copy_to_all else None

    # --- Bands plot + data
    bands_png = os.path.join(dirs["bands"], "bands_three_colors.png")
    bands_npz = os.path.join(dirs["data"], "bands_three_colors.npz")

    annotate = f"$N={N}$\n$U/t={U/t:.2f}$"
    plot_bands_three_colors(result, filename=bands_png, datafile=bands_npz, annotate_text=annotate)

    # --- DOS
    Egrid, dos = compute_dos(result, nE=dos_nE, eta=dos_eta)
    dos_png = os.path.join(dirs["dos"], "dos.png")
    dos_npz = os.path.join(dirs["data"], "dos.npz")
    plot_dos_styled(Egrid, dos, mu=result["mu"], filename=dos_png, datafile=dos_npz, annotate_text=annotate)

    # --- Magnetization
    mag_png = os.path.join(dirs["mag"], "mag_profile.png")
    mag_npz = os.path.join(dirs["data"], "mag_profile.npz")
    plot_magnetization_profile_styled(result, filename=mag_png, datafile=mag_npz, annotate_text=annotate)

    # --- Full raw summary for postprocessing
    summary_npz = os.path.join(dirs["data"], "run_summary.npz")
    np.savez(
        summary_npz,
        N=int(N), t=float(t), U=float(U), U_over_t=float(U / t),
        Nk=int(Nk), filling=float(filling),
        mu=float(result["mu"]),
        k_grid=result["k_grid"],
        E=result["E"],
        mA=result["mA"],
        mB=result["mB"],
        nA=result["nA"],
        nB=result["nB"],
        mix=float(mix), tol=float(tol), max_iter=int(max_iter), m0=float(m0),
        dos_nE=int(dos_nE), dos_eta=float(dos_eta),
    )

    # Copy to _ALL (GIF-ready)
    if copy_to_all:
        shutil.copy2(bands_png, os.path.join(all_dirs["bands"], f"bands_N{N:02d}_Ut{U/t:05.2f}.png"))
        shutil.copy2(dos_png, os.path.join(all_dirs["dos"], f"dos_N{N:02d}_Ut{U/t:05.2f}.png"))
        shutil.copy2(mag_png, os.path.join(all_dirs["mag"], f"mag_N{N:02d}_Ut{U/t:05.2f}.png"))

    return dirs["run"]


def run_sweep_N(
    *,
    N_min, N_max,
    t, U,
    a, Nk, filling, max_iter, mix, tol, m0,
    dos_nE, dos_eta,
    base_dir,
    mode_tag="sweep_N",
    verbose=False,
):
    all_dirs = make_all_collection_dirs(base_dir, mode_tag)
    for N in range(N_min, N_max + 1):
        print(f"\nRunning sweep_N: N={N}  U={U:.4f}  t={t:.4f}")
        out = run_single_case(
            N=N, t=t, U=U, a=a, Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir, mode_tag=mode_tag,
            copy_to_all=True, verbose=verbose,
        )
        print("Saved:", out)

    print("\nDONE sweep_N.")
    print("Collected images in:", os.path.join(base_dir, mode_tag, "_ALL"))
    print("  ", all_dirs["bands"])
    print("  ", all_dirs["dos"])
    print("  ", all_dirs["mag"])


def run_sweep_U(
    *,
    N_fixed,
    t,
    Umin_over_t, Umax_over_t, dU_over_t,
    a, Nk, filling, max_iter, mix, tol, m0,
    dos_nE, dos_eta,
    base_dir,
    mode_tag="sweep_U",
    verbose=False,
):
    all_dirs = make_all_collection_dirs(base_dir, mode_tag)

    U_over_t_list = np.arange(Umin_over_t, Umax_over_t + 1e-12, dU_over_t)
    for Ut in U_over_t_list:
        U = t * Ut
        print(f"\nRunning sweep_U: N={N_fixed}  U/t={Ut:.2f}  (U={U:.4f}, t={t:.4f})")
        out = run_single_case(
            N=N_fixed, t=t, U=U, a=a, Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir, mode_tag=mode_tag,
            copy_to_all=True, verbose=verbose,
        )
        print("Saved:", out)

    print("\nDONE sweep_U.")
    print("Collected images in:", os.path.join(base_dir, mode_tag, "_ALL"))
    print("  ", all_dirs["bands"])
    print("  ", all_dirs["dos"])
    print("  ", all_dirs["mag"])
