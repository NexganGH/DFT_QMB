# 04_fit_tb_to_dft_central.py
#
# Fit the non-interacting TB parameter t to the DFT π-bands
# of a zigzag graphene nanoribbon (ZGNR-N_ZIGZAG),
# using ONLY the two central π-bands (closest to EF).
#
# It then plots ONLY those two central DFT bands
# and their TB counterparts.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 0. Import ribbon parameters ---
from config_zgnr import N_ZIGZAG, LABEL_BASE

# --- 1. Import TB code from sibling folder "non interacting" ---

# Adjust path if your folder name is slightly different
this_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(this_dir, ".."))
tb_dir = os.path.join(parent_dir, "non interacting")
sys.path.append(tb_dir)

import dos_non_interacting as tb  # defines H_zgnr_k, etc.


# --- 2. Helper: TB bands on the same k-grid as DFT ---


def tb_bands_from_kdimless(k_dimless, N, t, a):
    """
    Compute TB bands E_n(k) for a given set of dimensionless
    k-values (k*a) matching the DFT grid.

    k_dimless : array (Nk,)
        Dimensionless k*a in [0, π].
    N         : int
        ZGNR width N_ZIGZAG.
    t         : float
        Nearest-neighbor hopping (eV), should be negative.
    a         : float
        Lattice spacing along ribbon (Å), same as a_dft.

    Returns
    -------
    E_tb : array (Nk, 2*N)
        Sorted eigenvalues at each k (in eV).
    """
    Nk = len(k_dimless)
    dim = 2 * N
    E_tb = np.zeros((Nk, dim), dtype=float)

    # convert dimensionless k*a back to k
    ks = k_dimless / a

    for i, k in enumerate(ks):
        Hk = tb.H_zgnr_k(k, N, t=t, a=a)
        w, _ = np.linalg.eigh(Hk)
        E_tb[i, :] = np.sort(w.real)

    return E_tb


def build_central_indices(n_pi, N_zigzag):
    """
    For 2*N_zigzag π-bands sorted at each k,
    return indices of the two central bands (closest to EF).

    Example: n_pi = 8 -> indices [3, 4]
    """
    assert n_pi == 2 * N_zigzag
    i1 = n_pi // 2 - 1  # highest occupied π band
    i2 = n_pi // 2      # lowest unoccupied π band
    return [i1, i2]


# --- 3. Cost function: only central bands ---


def cost(params, k_dimless, E_pi_dft, a_dft, N_zigzag, central_indices):
    """
    RMS difference between TB and DFT, using only
    the central π-bands specified by central_indices.
    """
    # enforce negative t; we parametrize via |t|
    t_raw = params[0]
    t = -abs(t_raw)

    E_tb = tb_bands_from_kdimless(k_dimless, N_zigzag, t, a_dft)

    # select central bands
    E_tb_c = E_tb[:, central_indices]       # (Nk, 2)
    E_dft_c = E_pi_dft[:, central_indices]  # (Nk, 2)

    diff = E_tb_c - E_dft_c
    rms = np.sqrt(np.mean(diff**2))
    return rms


# --- 4. Main fitting + plotting ---


def main():
    # --- 4.1 Load DFT π-bands ---
    fname = f"{LABEL_BASE}_pi_bands.npz"
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"{fname} not found. Run 03_select_pi_zgnr.py first."
        )

    data = np.load(fname)
    k_dimless = data["k_dimless"]   # (Nk,)
    E_pi_dft = data["E_pi_dft"]     # (Nk, 2*N_ZIGZAG)
    a_dft = data["a_dft"]           # lattice constant along ribbon (Å)

    Nk, n_pi = E_pi_dft.shape
    print(f"Loaded DFT π-bands from {fname}")
    print(f"  Nk   = {Nk}")
    print(f"  n_pi = {n_pi} (expected 2*N_ZIGZAG = {2 * N_ZIGZAG})")

    if n_pi != 2 * N_ZIGZAG:
        print("WARNING: n_pi != 2*N_ZIGZAG, check inputs!")

    # --- 4.2 Identify central π bands ---
    central_indices = build_central_indices(n_pi, N_ZIGZAG)
    print("Central π-band indices (0-based):", central_indices)

    # --- 4.3 Fit t using only central bands ---
    x0 = np.array([-2.7])  # initial guess in eV

    res = minimize(
        cost,
        x0,
        args=(k_dimless, E_pi_dft, a_dft, N_ZIGZAG, central_indices),
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
    )

    t_fit = -abs(res.x[0])
    print("\n=== Fit result (central bands only) ===")
    print(f"  t_fit       = {t_fit:.4f} eV")
    print(f"  RMS error   = {res.fun:.4f} eV")
    print(f"  Converged   = {res.success}")
    print(f"  Message     = {res.message}")

    # --- 4.4 Compute TB bands with fitted t (for plotting) ---
    E_tb = tb_bands_from_kdimless(k_dimless, N_ZIGZAG, t_fit, a_dft)

    # --- 4.5 Plot ONLY the central fitted π bands ---
    fig, ax = plt.subplots(figsize=(7, 5))

    # DFT central π bands (solid)
    ax.plot(
        k_dimless / np.pi,
        E_pi_dft[:, central_indices[0]],
        color="C0",
        lw=2.2,
        label="DFT π (valence)",
    )
    ax.plot(
        k_dimless / np.pi,
        E_pi_dft[:, central_indices[1]],
        color="C1",
        lw=2.2,
        label="DFT π (conduction)",
    )

    # TB central π bands (dashed)
    ax.plot(
        k_dimless / np.pi,
        E_tb[:, central_indices[0]],
        "--",
        color="C0",
        lw=2.2,
        label="TB fit (valence)",
    )
    ax.plot(
        k_dimless / np.pi,
        E_tb[:, central_indices[1]],
        "--",
        color="C1",
        lw=2.2,
        label="TB fit (conduction)",
    )

    ax.axhline(0.0, ls="--", lw=0.7, color="k", alpha=0.6)

    ax.set_xlabel(r"$k a / \pi$")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"ZGNR-{N_ZIGZAG}: Central π Bands – DFT vs TB Fit")

    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    out_png = f"{LABEL_BASE}_pi_fit_central_ONLY.png"
    plt.savefig(out_png, dpi=200)
    plt.show()

    print(f"\nSaved clean central-band plot as {out_png}")


if __name__ == "__main__":
    main()
