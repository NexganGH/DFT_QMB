import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Importa solve_zgnr_mf dal tuo file TB
import sys
import os

# add parent directory to python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from mft.magnetisation_mft import solve_zgnr_mf


# ============================================================
# 1. Load DFT data and convert k → dimensionless ka ∈ [0, π]
# ============================================================

def load_dft_pi_data(npz_file="zgnr_pi_bands.npz"):
    """Load k (converted to ka) and all DFT bands (relative to EF) from npz.

    ALWAYS returns:
        k_dft: dimensionless ka ∈ [0, π]
        E_rel_all: shape (nspins, Nk, nbands)
        data: raw npz object (for bands_up / bands_dn, etc.)
    """

    data = np.load(npz_file)
    print("Available keys in npz:", data.files)

    # 1) Choose raw k-axis from file
    if "k_dimless" in data.files:
        k_raw = data["k_dimless"]
        source = "k_dimless"
    elif "k" in data.files:
        k_raw = data["k"]
        source = "k"
    elif "k_dist" in data.files:
        k_raw = data["k_dist"]
        source = "k_dist"
    else:
        raise KeyError(
            "No suitable k-array found in npz. Expected one of: "
            "'k_dimless', 'k', 'k_dist'. Found keys: "
            + ", ".join(data.files)
        )

    print(f"Using '{source}' from npz as raw k-axis.")
    print("Raw k range:", k_raw[0], "→", k_raw[-1])

    # 2) Ensure we have dimensionless ka in [0, π]
    k_min = float(k_raw.min())
    k_max = float(k_raw.max())

    if (k_min >= -1e-6
        and np.isclose(k_max, np.pi, rtol=1e-2, atol=1e-2)):
        # Already looks like [0, π]
        k_dft = k_raw
        print("Interpreting raw k as dimensionless ka in [0, π].")
    else:
        # Old style: k_raw is path distance Γ→X in 1/Å
        # Rescale so last point maps to π
        k_dft = k_raw / k_max * np.pi
        print("Rescaled raw k to dimensionless ka in [0, π].")
        print("Rescaling factor:", np.pi / k_max)

    print("Final k_dft (ka) range:", k_dft[0], "→", k_dft[-1])

    E_rel_all = data["E_rel_all"]  # (nspins, Nk, nbands)
    return k_dft, E_rel_all, data


# ============================================================
# 2. Band selection on DFT side
# ============================================================

def select_dft_pi_bands_fixed_indices(E_rel_all, band_indices, spin_index=1):
    """
    Select *fixed* DFT band indices for one spin.

    band_indices: iterable of ints (global band indices in E_rel_all)
    Returns E_dft_sel of shape (Nk, len(band_indices)).
    """
    E_spin = E_rel_all[spin_index]       # (Nk, nbands)
    band_indices = np.array(band_indices, dtype=int)
    E_sel = E_spin[:, band_indices]      # (Nk, n_bands_keep)
    return E_sel


def select_dft_pi_bands(E_rel_all, spin_index=1, n_bands_keep=2,
                        E_window=(-3.0, 3.0)):
    """
    Fallback: per-k selection of the n_bands_keep bands (for one spin)
    closest to E=0 inside window E_window.

    WARNING: this can change band identity along k and create
    artificial vertical jumps if used alone.
    """
    E_min, E_max = E_window
    E_spin = E_rel_all[spin_index]      # (Nk, nbands)
    Nk, nbands = E_spin.shape

    E_sel = np.zeros((Nk, n_bands_keep))
    for ik in range(Nk):
        row = E_spin[ik, :]
        mask = (row > E_min) & (row < E_max)
        candidates = row[mask]
        if len(candidates) < n_bands_keep:
            idx_sort = np.argsort(np.abs(row))
            chosen = idx_sort[:n_bands_keep]
            vals = row[chosen]
        else:
            idx_range = np.where(mask)[0]
            vals_range = row[idx_range]
            order = np.argsort(np.abs(vals_range))
            chosen = idx_range[order[:n_bands_keep]]
            vals = row[chosen]

        E_sel[ik, :] = np.sort(vals)

    return E_sel


# ============================================================
# 3. TB bands on same k-grid as DFT
# ============================================================

def compute_tb_bands_on_k(Ny, t, U, k_dft, filling=1.0,
                          Nk_tb=400, max_iter=200):
    """
    Compute TB MF bands with solve_zgnr_mf, then interpolate
    the first 2 spin-down bands onto the DFT k-grid (ka ∈ [0, π]).

    Returns: E_tb_interp shape (Nk_dft, 2)
    """
    result_tb = solve_zgnr_mf(
        Ny=Ny,
        U=U,
        t=t,
        a=1.0,
        Nk=Nk_tb,
        filling=filling,
        max_iter=max_iter,
        mix=0.1,
        tol=1e-5,
        verbose=False,
    )

    k_tb_full = result_tb["k_grid"]          # [-π, π)
    E_tb_full = result_tb["E"][1, :, :]      # spin down, shape (Nk_tb, dim)
    mu_tb = result_tb["mu"]

    # energies relative to TB chemical potential
    E_tb_full_rel = E_tb_full - mu_tb       # (Nk_tb, dim)

    # Keep only k ≥ 0 side for interpolation onto [0, π]
    mask_pos = k_tb_full >= 0.0
    k_tb_pos = k_tb_full[mask_pos]
    E_tb_pos = E_tb_full_rel[mask_pos, :]   # (Nk_pos, dim)

    # take first two bands (assume these are the relevant π-like bands)
    n_bands_tb = 2
    E_tb_pos_sel = E_tb_pos[:, :n_bands_tb]

    # Interpolate onto DFT k-grid
    E_tb_interp = np.zeros((len(k_dft), n_bands_tb))
    for ib in range(n_bands_tb):
        E_tb_interp[:, ib] = np.interp(k_dft, k_tb_pos, E_tb_pos_sel[:, ib])

    return E_tb_interp


# ============================================================
# 4. Loss function and fit
# ============================================================

def loss_t_u(params, Ny, k_dft, E_dft_sel):
    """Mean squared error between DFT and TB bands."""
    t, U = params
    if t <= 0 or U <= 0:
        return 1e6

    try:
        E_tb_interp = compute_tb_bands_on_k(
            Ny=Ny,
            t=t,
            U=U,
            k_dft=k_dft,
            filling=1.0,
            Nk_tb=400,
            max_iter=200,
        )
    except Exception as e:
        print("Warning: solve_zgnr_mf failed for t,U =", t, U, "err:", e)
        return 1e6

    if E_tb_interp.shape != E_dft_sel.shape:
        print("Shape mismatch in loss:", E_tb_interp.shape, E_dft_sel.shape)
        return 1e6

    diff = E_tb_interp - E_dft_sel
    return np.mean(diff**2)


def fit_tb_to_dft(Ny_fit=2,
                  npz_file="zgnr_pi_bands.npz",
                  spin_index=1,
                  n_bands_keep=2,
                  E_window=(-3.0, 3.0)):
    """
    Fit (t, U) to the DFT π-bands saved in npz_file.

    Ny_fit: number of zigzag chains in TB model (match DFT ribbon: 2 here).
    spin_index: 0 = spin up, 1 = spin down (GPAW convention).
    """

    print("=== FIT TB ↔ DFT π-bands ===")
    print(f"Ny(TB) = {Ny_fit}, npz_file = {npz_file}")

    # Load DFT data
    k_dft, E_rel_all, data = load_dft_pi_data(npz_file=npz_file)
    Nk_dft = len(k_dft)
    print(f"Loaded DFT data: Nk_dft = {Nk_dft}, shape E_rel_all = {E_rel_all.shape}")

    # Try to use the π-band indices stored in the npz
    if "bands_dn" in data.files and spin_index == 1:
        bands_dn = data["bands_dn"]
    elif "bands_up" in data.files and spin_index == 0:
        bands_dn = data["bands_up"]
    else:
        bands_dn = None

    if bands_dn is not None and len(bands_dn) >= n_bands_keep:
        band_indices = np.array(bands_dn[:n_bands_keep], dtype=int)
        print("Using fixed DFT band indices:", band_indices)
        E_dft_sel = select_dft_pi_bands_fixed_indices(
            E_rel_all,
            band_indices=band_indices,
            spin_index=spin_index,
        )
    else:
        print("No suitable 'bands_dn'/'bands_up' in npz; "
              "falling back to per-k selection.")
        E_dft_sel = select_dft_pi_bands(
            E_rel_all,
            spin_index=spin_index,
            n_bands_keep=n_bands_keep,
            E_window=E_window,
        )

    print("Selected DFT bands shape:", E_dft_sel.shape)

    # Initial guess
    x0 = np.array([2.7, 2.0])   # guess: t ~ 2.7 eV, U ~ 2 eV
    print("Initial guess: t = %.3f eV, U = %.3f eV" % (x0[0], x0[1]))

    # Minimize loss
    res = minimize(
        loss_t_u,
        x0,
        args=(Ny_fit, k_dft, E_dft_sel),
        method="Nelder-Mead",
        options={"maxiter": 30, "disp": True},
    )

    t_opt, U_opt = res.x
    print("\n=== FIT COMPLETED ===")
    print("Optimized parameters:")
    print(f"  t = {t_opt:.4f} eV")
    print(f"  U = {U_opt:.4f} eV")
    print("Final loss (mean squared error) =", res.fun)
    print("======================\n")

    # Compute TB bands with optimal parameters
    E_tb_best = compute_tb_bands_on_k(
        Ny=Ny_fit,
        t=t_opt,
        U=U_opt,
        k_dft=k_dft,
        filling=1.0,
        Nk_tb=400,
        max_iter=200,
    )

    # Plot comparison
    plt.figure(figsize=(5, 6))
    for ib in range(E_dft_sel.shape[1]):
        plt.plot(k_dft, E_dft_sel[:, ib],
                 color=f"C{ib}", ls="-", lw=1.2,
                 label=f"DFT band {ib+1}")
    for ib in range(E_tb_best.shape[1]):
        plt.plot(k_dft, E_tb_best[:, ib],
                 color=f"C{ib}", ls="--", lw=1.2,
                 label=f"TB fit band {ib+1}")

    plt.axhline(0.0, ls="--", color="k", alpha=0.6)
    plt.xlabel(r"$ka$")
    plt.ylabel(r"$E - E_F$ (eV)")
    plt.xticks([0.0, np.pi], [r"$0$", r"$\pi$"])
    plt.title("DFT vs TB (fitted) π-bands, spin-down")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dft_tb_pi_fit.png", dpi=300)
    plt.close()
    print("Saved comparison plot to 'dft_tb_pi_fit.png'")

    return t_opt, U_opt


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    # Match the DFT ribbon: 2 zigzag stripes
    Ny_fit = 2

    t_fit, U_fit = fit_tb_to_dft(
        Ny_fit=Ny_fit,
        npz_file="zgnr_pi_bands.npz",
        spin_index=1,          # spin-down
        n_bands_keep=2,
        E_window=(-3.0, 3.0),
    )
    print(f"\nBest-fit t = {t_fit:.4f} eV, U = {U_fit:.4f} eV")
