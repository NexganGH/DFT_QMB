import numpy as np
import matplotlib.pyplot as plt

# ---------- same H_zgnr_k and bands_zgnr as before ----------

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


def bands_zgnr(N=6, nk=400, t=-2.7, a=1.0):
    ks = np.linspace(-np.pi/a, np.pi/a, nk)
    n_bands = 2 * N
    energies = np.zeros((nk, n_bands), dtype=float)

    for i, k in enumerate(ks):
        Hk = H_zgnr_k(k, N, t=t, a=a)
        w, _ = np.linalg.eigh(Hk)
        energies[i, :] = np.sort(w.real)

    return ks, energies


# ---------- 2. DOS from eigenvalues + Gaussian broadening ----------

def dos_from_bands(energies, n_E=1000, eta=0.05):
    """
    Compute DOS(E) with Gaussian broadening from an array of energies.

    energies : array (nk, nbands)
        Eigenvalues in energy units of |t|.
    n_E : int
        Number of points in the energy grid.
    eta : float
        Gaussian broadening (standard deviation) in same units as energies.

    Returns
    -------
    E_grid : array (n_E,)
    DOS    : array (n_E,)
        DOS per unit cell.
    """
    # Flatten all eigenvalues (nk * nbands)
    eps = energies.ravel()
    Emin, Emax = eps.min(), eps.max()

    # Extend slightly beyond min/max to capture tails
    pad = 0.2 * (Emax - Emin)
    Emin -= pad
    Emax += pad

    E_grid = np.linspace(Emin, Emax, n_E)

    # Gaussian broadening
    # ρ(E) = (1/N_k) * sum_{k,n} g_eta(E - eps_{k,n})
    # g_eta(x) = 1/(sqrt(2π) η) exp(-x^2/(2η^2))
    # factor 1/(N_k) makes DOS per unit cell
    nk = energies.shape[0]
    x = E_grid[:, None] - eps[None, :]
    gaussians = np.exp(-0.5 * (x / eta)**2) / (np.sqrt(2*np.pi) * eta)
    DOS = gaussians.sum(axis=1) / nk

    return E_grid, DOS


# ---------- 3. Example: band structure + DOS for one width ----------

def plot_bands_and_dos(N=20, nk=1000, t=-2.7, a=1.0, eta=0.05):
    ks, E = bands_zgnr(N=N, nk=nk, t=t, a=a)
    E_grid, DOS = dos_from_bands(E, n_E=1000, eta=eta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios':[2,1]})

    # --- Band structure ---
    for n in range(E.shape[1]):
        ax1.plot(ks/np.pi, E[:, n], lw=1)
    ax1.axhline(0.0, ls='--', lw=0.8)
    ax1.set_xlabel(r"$k/\pi$")
    ax1.set_ylabel("Energy (|t|)")
    ax1.set_title(f"ZGNR bands, N = {N}")

    # --- DOS ---
    ax2.plot(DOS, E_grid, lw=1)
    ax2.axhline(0.0, ls='--', lw=0.8)
    ax2.set_xlabel("DOS (states / |t| / cell)")
    ax2.set_ylabel("Energy (|t|)")
    ax2.set_title("DOS")
    ax2.set_ylim(E_grid.min(), E_grid.max())
    ax2.grid(True, ls=':', alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # choose ribbon width and broadening
    plot_bands_and_dos(N=60, nk=1000, eta=0.05)
