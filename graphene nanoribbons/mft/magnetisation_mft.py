import numpy as np
import matplotlib.pyplot as plt


###ATTENTION in this code the total number of strands is given by Ny but the index that labels them
### is m and not n as in teh theory. In the theory m stands for the primitive cell we are considering along
### the x direction and n stands for the label of the strands.
###However here the code is correct and works properly

# ----------------------------------------------------------
# 1. Tight-binding Hamiltonian H0(k) for zigzag nanoribbon
# ----------------------------------------------------------

def build_h0_zigzag(Ny, k, t=1.0, a=1.0):
    """
    2*Ny x 2*Ny tight-binding Hamiltonian for a zigzag nanoribbon.
    Basis: [A1, B1, A2, B2, ..., A_Ny, B_Ny].
    """
    dim = 2 * Ny
    H0 = np.zeros((dim, dim), dtype=np.complex128)

    for l in range(dim - 1):
        if l % 2 == 0:
            # A_m -> B_m, amplitude -2 t cos(ka/2)
            val = -2.0 * t * np.cos(k * a / 2.0)
        else:
            # B_m -> A_{m+1}, amplitude -t
            val = -t

        H0[l, l + 1] = val
        H0[l + 1, l] = np.conjugate(val)

    return H0


# ----------------------------------------------------------
# 2. Mean-field interaction: diagonal spin-dependent matrix
# ----------------------------------------------------------

def build_V_sigma(Ny, sigma, U, mA, mB):
    """
    Diagonal 2*Ny x 2*Ny matrix for mean-field Hubbard:
      H_U^MF ≈ -U Σ_{m,σ} σ [ mA[m] n_{A,m,σ} + mB[m] n_{B,m,σ} ].
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


# ----------------------------------------------------------
# 3. Self-consistent mean-field solver
# ----------------------------------------------------------

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
    verbose=True,
):
    """
    Self-consistent mean-field Hubbard solution for zigzag nanoribbon.
    Returns eigenvalues, eigenvectors, magnetizations, etc.

    filling = 1.0 -> one electron per site (half filling, since max is 2).
    """
    dim = 2 * Ny

    # k-grid along 1D BZ
    k_min = -np.pi / a
    k_max = np.pi / a
    k_grid = np.linspace(k_min, k_max, Nk, endpoint=False)

    # Number of sites per cell = 2*Ny
    n_sites = dim
    # electrons per *cell* at given filling (1.0 => 1 electron per site)
    nelec_per_cell = filling * n_sites  # e.g. 2*Ny at half-filling

    if verbose:
        print("----------------------------------------------------")
        print(f"solve_zgnr_mf: Ny = {Ny}, U/t = {U/t:.3f}, Nk = {Nk}")
        print(f"Target electrons per cell ≈ {nelec_per_cell:.1f}")
        print("Initial AFM edge guess for mA, mB")
        print("----------------------------------------------------")

    # Initial AFM edge guess
    mA = np.zeros(Ny)
    mB = np.zeros(Ny)
    m0 = 0.05
    mA[0] = +m0
    mB[0] = +m0
    mA[-1] = -m0
    mB[-1] = -m0

    if verbose:
        print("Initial mA:", mA)
        print("Initial mB:", mB)

    # Storage
    E = np.zeros((2, Nk, dim), dtype=np.float64)              # [spin, k, band]
    Uvec = np.zeros((2, Nk, dim, dim), dtype=np.complex128)   # eigenvectors
    mu = 0.0

    for it in range(max_iter):
        # 1. Diagonalise H_sigma(k) for current mA, mB
        for s_idx, sigma in enumerate([+1, -1]):  # 0: up, 1: down
            for ik, k in enumerate(k_grid):
                H0 = build_h0_zigzag(Ny, k, t=t, a=a)
                V = build_V_sigma(Ny, sigma, U, mA, mB)
                H = H0 + V
                vals, vecs = np.linalg.eigh(H)
                E[s_idx, ik, :] = vals
                Uvec[s_idx, ik, :, :] = vecs

        # 2. Find Fermi level at T=0 by filling nelec_per_cell * Nk lowest states
        E_flat = E.reshape(-1)
        idx_sorted = np.argsort(E_flat)

        total_states = len(E_flat)
        nelec_total = int(round(nelec_per_cell * Nk))  # total electrons in all k-cells

        if nelec_total >= total_states:
            mu = E_flat[idx_sorted[-1]] + 1e-6
        else:
            eN_1 = E_flat[idx_sorted[nelec_total - 1]]
            eN = E_flat[idx_sorted[nelec_total]]
            mu = 0.5 * (eN_1 + eN)

        # 3. Compute local densities from occupied states
        nA_up = np.zeros(Ny)
        nA_dn = np.zeros(Ny)
        nB_up = np.zeros(Ny)
        nB_dn = np.zeros(Ny)

        for s_idx, sigma in enumerate([+1, -1]):
            for ik in range(Nk):
                vals = E[s_idx, ik, :]
                vecs = Uvec[s_idx, ik, :, :]
                occ = vals <= mu
                occ_inds = np.where(occ)[0]
                w_k = 1.0 / Nk

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

        # 4. Mixing
        mA_mixed = (1.0 - mix) * mA + mix * mA_new
        mB_mixed = (1.0 - mix) * mB + mix * mB_new

        delta_m = max(
            np.max(np.abs(mA_mixed - mA)),
            np.max(np.abs(mB_mixed - mB))
        )
        mA, mB = mA_mixed, mB_mixed

        if verbose:
            print(
                f"Iter {it+1:3d}: mu = {mu:+.6f}, "
                f"max|Δm| = {delta_m:.3e}, "
                f"edge mA = ({mA[0]:+.4f}, {mA[-1]:+.4f})"
            )

        if delta_m < tol:
            if verbose:
                print("Converged.")
            break

    # Explicit prints *inside* the solver:
    if verbose:
        print("----------------------------------------------------")
        print("Final magnetization profile from solve_zgnr_mf:")
        print("mA:", mA)
        print("mB:", mB)
        print("Final Fermi level mu =", mu)
        print("----------------------------------------------------")

        # Also print total average density as a sanity check
        avg_nA = np.mean(nA_new)
        avg_nB = np.mean(nB_new)
        print(f"Average nA per site: {avg_nA:.4f}")
        print(f"Average nB per site: {avg_nB:.4f}")
        print(f"Average total density per site: {(avg_nA + avg_nB)/2:.4f}")
        print("----------------------------------------------------")

    result = {
        "k_grid": k_grid,
        "E": E,
        "Uvec": Uvec,
        "mA": mA,
        "mB": mB,
        "nA": nA_new,
        "nB": nB_new,
        "mu": mu,
        "t": t,
        "U": U,
        "Ny": Ny,
    }
    return result


# ----------------------------------------------------------
# 4. DOS computation
# ----------------------------------------------------------

def compute_dos(result, nE=400, E_min=None, E_max=None, eta=0.05):
    """
    Compute DOS using Gaussian broadening of eigenvalues.

    Parameters
    ----------
    result : dict
        Output from solve_zgnr_mf.
    nE : int
        Number of energy points.
    E_min, E_max : float or None
        Energy window for DOS. If None, inferred from eigenvalues.
    eta : float
        Broadening (Gaussian width).

    Returns
    -------
    Egrid : (nE,) array
    dos   : (nE,) array
    """
    Evals = result["E"]  # shape (2, Nk, dim)
    k_grid = result["k_grid"]
    Nk = len(k_grid)

    E_flat = Evals.reshape(-1)

    if E_min is None:
        E_min = E_flat.min() - 3 * eta
    if E_max is None:
        E_max = E_flat.max() + 3 * eta

    Egrid = np.linspace(E_min, E_max, nE)
    dos = np.zeros_like(Egrid)

    # Weight per k and per spin is 1 / Nk
    w_k = 1.0 / Nk

    # Gaussian broadening
    pref = 1.0 / (np.sqrt(2 * np.pi) * eta)

    for E0 in E_flat:
        dos += w_k * pref * np.exp(-0.5 * ((Egrid - E0) / eta) ** 2)

    return Egrid, dos


# ----------------------------------------------------------
# 5. Plotting helpers (and saving to files)
# ----------------------------------------------------------

def plot_bands(result, spin_resolved=True, filename=None):
    """
    Plot band structure and optionally save to file.
    """
    k = result["k_grid"]
    E = result["E"]
    mu = result["mu"]
    dim = E.shape[-1]

    plt.figure()
    if spin_resolved:
        for s_idx, label in zip([0, 1], ["↑", "↓"]):
            for b in range(dim):
                plt.plot(k, E[s_idx, :, b], lw=0.8, label=label if b == 0 else "")
    else:
        for s_idx in [0, 1]:
            for b in range(dim):
                plt.plot(k, E[s_idx, :, b], lw=0.8)

    plt.axhline(mu, ls="--", alpha=0.5)
    plt.xlabel("k")
    plt.ylabel("Energy")
    if spin_resolved:
        plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()


def plot_dos(Egrid, dos, mu=0.0, filename=None):
    """
    Plot DOS vs energy and optionally save to file.
    """
    plt.figure()
    plt.plot(Egrid, dos)
    plt.axvline(mu, ls="--", alpha=0.5)
    plt.xlabel("Energy")
    plt.ylabel("DOS (arb. units)")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()


def plot_magnetization_profile(result, filename=None):
    """
    Plot mA[m] and mB[m] vs strand index m and optionally save.
    """
    mA = result["mA"]
    mB = result["mB"]
    Ny = result["Ny"]
    strands = np.arange(Ny)

    plt.figure()
    plt.plot(strands, mA, marker="o", label="mA")
    plt.plot(strands, mB, marker="s", label="mB")
    plt.xlabel("Strand index m")
    plt.ylabel("Magnetization m_m")
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.close()


# ----------------------------------------------------------
# 6. Example usage & explicit prints
# ----------------------------------------------------------

if __name__ == "__main__":
    # Choose number of strands Ny:
    # Ny = 4, 8, 16, ... larger Ny = wider ribbon (more 1D-like edges)
    Ny = 1
    t = 1.0
    U = 1     # try 2.0, 2.5, 3.0 to see stronger magnetization
    Nk = 400
    filling = 1.0   # half-filling: 1 electron per site

    print("====================================================")
    print("Running MF Hubbard ZGNR calculation (main section)")
    print(f"Ny = {Ny}, U = {U}, t = {t}, Nk = {Nk}, filling = {filling}")
    print("====================================================")

    result = solve_zgnr_mf(
        Ny=Ny,
        U=U,
        t=t,
        a=1.0,
        Nk=Nk,
        filling=filling,
        max_iter=300,
        mix=0.1,
        tol=1e-5,
        verbose=True,
    )

    # Explicit prints here as well
    print("\n================ FINAL RESULTS (from __main__) ================")
    print("Final edge magnetizations:")
    print("mA[0], mA[-1] =", result["mA"][0], result["mA"][-1])
    print("mB[0], mB[-1] =", result["mB"][0], result["mB"][-1])
    print("\nFull mA array:", result["mA"])
    print("Full mB array:", result["mB"])
    print("Final Fermi level mu:", result["mu"])
    print("====================================================\n")

    # --- Bands ---
    bands_fname = f"bands_Ny{Ny}_U{U:.2f}.png"
    plot_bands(result, spin_resolved=True, filename=bands_fname)
    print(f"Band structure saved to {bands_fname}")

    # --- DOS ---
    Egrid, dos = compute_dos(result, nE=600, eta=0.05)
    dos_fname = f"dos_Ny{Ny}_U{U:.2f}.png"
    plot_dos(Egrid, dos, mu=result["mu"], filename=dos_fname)
    print(f"DOS saved to {dos_fname}")

    # --- Magnetization profile ---
    mag_fname = f"mag_profile_Ny{Ny}_U{U:.2f}.png"
    plot_magnetization_profile(result, filename=mag_fname)
    print(f"Magnetization profile saved to {mag_fname}")
    print("====================================================")
