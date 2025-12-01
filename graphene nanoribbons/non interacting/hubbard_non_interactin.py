import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Hamiltonian H(k) for a ZGNR of width N ----------

def H_zgnr_k(k, N, t=-1.0, a=1.0):
    """
    Tight-binding Hamiltonian H(k) for a zigzag graphene nanoribbon
    of width N (N zigzag chains), nearest-neighbour only.

    Basis: [A1, B1, A2, B2, ..., A_N, B_N]

    Parameters
    ----------
    k : float
        1D crystal momentum along ribbon (in units where a=1 if you like)
    N : int
        Number of zigzag chains (width)
    t : float
        Hopping (default -1.0, sign is conventional)
    a : float
        Lattice period along x (set to 1.0; just rescales k)
    """
    alpha = 2.0 * np.cos(k * a / 2.0)
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=complex)

    for n in range(1, N + 1):
        iA = 2 * (n - 1)     # index of A_n  (0-based)
        iB = 2 * (n - 1) + 1 # index of B_n

        # A_n <-> B_n  (horizontal bond, k-dependent)
        H[iA, iB] += t * alpha
        H[iB, iA] += t * alpha

        # A_n <-> B_{n-1}  (diagonal downwards, only if n > 1)
        if n > 1:
            iB_prev = 2 * (n - 2) + 1
            H[iA, iB_prev] += t
            H[iB_prev, iA] += t

        # B_n <-> A_{n+1}  (diagonal upwards, only if n < N)
        if n < N:
            iA_next = 2 * n
            H[iB, iA_next] += t
            H[iA_next, iB] += t

    return H


# ---------- 2. Compute band structure ----------

def bands_zgnr(N=6, nk=200, t=-1.0, a=1.0):
    """
    Compute band structure E_n(k) for a ZGNR of width N.
    Returns ks, energies (nk x 2N).
    """
    ks = np.linspace(-np.pi/a, np.pi/a, nk)
    n_bands = 2 * N
    energies = np.zeros((nk, n_bands), dtype=float)

    for i, k in enumerate(ks):
        Hk = H_zgnr_k(k, N, t=t, a=a)
        w, _ = np.linalg.eigh(Hk)
        energies[i, :] = np.sort(w.real)

    return ks, energies


# ---------- 3. Plot example for different widths ----------

def plot_zgnr(N_list=(1, 4, 8), nk=300, t=-1.0, a=1.0):
    for N in N_list:
        ks, E = bands_zgnr(N=N, nk=nk, t=t, a=a)

        plt.figure(figsize=(5, 4))
        for n in range(E.shape[1]):
            plt.plot(ks / np.pi, E[:, n], lw=1)

        plt.axhline(0.0, ls='--', lw=0.8)  # Fermi level at half filling
        plt.xlabel(r"$k / \pi$")
        plt.ylabel("Energy (units of |t|)")
        plt.title(f"ZGNR, width N = {N} (2N = {2*N} bands)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Change N_list to the widths you want to compare
    plot_zgnr(N_list=(1, 4, 8), nk=400)
