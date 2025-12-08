import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# 0. Import your TB solver without touching magnetisation_mft.py
# ============================================================

this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
sys.path.append(project_root)

from mft.magnetisation_mft import solve_zgnr_mf


# ============================================================
# 1. Helpers to extract smooth π bands from E(k,n)
# ============================================================

def choose_start_indices(E):
    Nk, nbands = E.shape
    i0 = Nk // 2
    Ek = E[i0]

    mask_val = Ek < 0.0
    if not np.any(mask_val):
        raise RuntimeError("No energies below EF at chosen k.")
    val_candidates = np.where(mask_val)[0]
    n_val = val_candidates[np.argmax(Ek[mask_val])]

    mask_cond = Ek > 0.0
    if not np.any(mask_cond):
        raise RuntimeError("No energies above EF at chosen k.")
    cond_candidates = np.where(mask_cond)[0]
    n_cond = cond_candidates[np.argmin(Ek[mask_cond])]

    return i0, n_val, n_cond


def track_band_by_continuity(E, i0, n0):
    Nk, nbands = E.shape
    idx = np.empty(Nk, dtype=int)
    idx[i0] = n0

    for i in range(i0 + 1, Nk):
        E_prev = E[i - 1, idx[i - 1]]
        diffs = np.abs(E[i] - E_prev)
        idx[i] = np.argmin(diffs)

    for i in range(i0 - 1, -1, -1):
        E_prev = E[i + 1, idx[i + 1]]
        diffs = np.abs(E[i] - E_prev)
        idx[i] = np.argmin(diffs)

    return idx


def extract_smooth_pi(E_rel):
    Nk, nbands = E_rel.shape
    i0, n_val0, n_cond0 = choose_start_indices(E_rel)
    idx_val = track_band_by_continuity(E_rel, i0, n_val0)
    idx_cond = track_band_by_continuity(E_rel, i0, n_cond0)
    E_val = E_rel[np.arange(Nk), idx_val]
    E_cond = E_rel[np.arange(Nk), idx_cond]
    return E_val, E_cond


# ============================================================
# 2. Load smoothed DFT π bands
# ============================================================

dft_file = "zgnr_pi_smooth_from_Z.npz"
data_dft = np.load(dft_file)

k_dft = data_dft["k_dimless"]    # (Nk,)
E_val_dft = data_dft["E_val"]    # (Nk,)
E_cond_dft = data_dft["E_cond"]  # (Nk,)

Nk_dft = k_dft.size
print(f"Loaded DFT smooth π bands from {dft_file}, Nk = {Nk_dft}")

# must match the DFT ribbon
Ny = 4


# ============================================================
# 3. Wrapper around solve_zgnr_mf
# ============================================================

def tb_bands_and_magnetization(k_dft, t, U, Ny):
    """
    Run your MF TB model and return TB π val/cond on the same
    k-grid as the DFT (Γ→Z, 0..π).
    """
    Nk_tb = 2 * len(k_dft)  # k from -π..π

    result = solve_zgnr_mf(
        Ny=Ny,
        U=U,
        t=t,
        a=1.0,
        Nk=Nk_tb,
        filling=1.0,
        max_iter=200,
        mix=0.1,
        tol=1e-5,
        verbose=False,
    )

    E_all = result["E"]          # (2, Nk_tb, dim)
    k_tb = result["k_grid"]      # (Nk_tb,)
    mu_tb = result["mu"]

    # spin-averaged bands
    E_mean = 0.5 * (E_all[0] + E_all[1])  # (Nk_tb, dim)

    # keep k >= 0 to mimic Γ→Z
    idx_pos = np.where(k_tb >= 0.0)[0]
    if len(idx_pos) != len(k_dft):
        raise RuntimeError(
            f"TB positive-k points = {len(idx_pos)} != Nk_dft = {len(k_dft)}"
        )

    E_mean_pos = E_mean[idx_pos, :]      # (Nk_dft, dim)
    E_rel_tb = E_mean_pos - mu_tb        # relative to EF

    E_val_tb, E_cond_tb = extract_smooth_pi(E_rel_tb)
    return E_val_tb, E_cond_tb, result


# ============================================================
# 4. Cost function for fitting t and U
# ============================================================

def objective(params):
    t, U = params
    E_val_tb, E_cond_tb, _ = tb_bands_and_magnetization(k_dft, t, U, Ny)

    err_val = E_val_tb - E_val_dft
    err_cond = E_cond_tb - E_cond_dft
    err = np.concatenate([err_val, err_cond])
    rms = np.sqrt(np.mean(err**2))
    print(f"t = {t:.3f}, U = {U:.3f}  ->  RMS = {rms:.4f} eV")
    return rms


# ============================================================
# 5. Fit t and U
# ============================================================

t0 = 2.7
U0 = 3.0

res = minimize(
    objective,
    x0=np.array([t0, U0]),
    method="Nelder-Mead",
    options=dict(maxiter=100, xatol=1e-3, fatol=1e-4, disp=True),
)

t_fit, U_fit = res.x
print("\n====================")
print("Best fit parameters:")
print(f"t = {t_fit:.4f} eV")
print(f"U = {U_fit:.4f} eV")
print("====================\n")


# ============================================================
# 6. Recompute TB at best-fit and plot
# ============================================================

E_val_tb, E_cond_tb, result_tb = tb_bands_and_magnetization(k_dft, t_fit, U_fit, Ny)

# --- bands ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(k_dft/np.pi, E_val_dft,  lw=2.5, color="C1", label="DFT π valence")
ax.plot(k_dft/np.pi, E_cond_dft, lw=2.5, color="C0", label="DFT π conduction")
ax.plot(k_dft/np.pi, E_val_tb,  "--", lw=2.0, color="C3", label="TB π valence (fit)")
ax.plot(k_dft/np.pi, E_cond_tb, "--", lw=2.0, color="C2", label="TB π conduction (fit)")
ax.axhline(0.0, ls="--", lw=0.8, color="k", alpha=0.6)
ax.set_xlabel(r"$k a/\pi$")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title(r"Central $\pi$ bands: DFT vs TB fit")
ax.legend()
fig.tight_layout()
fig.savefig("pi_bands_dft_vs_tb_fit.png", dpi=300)
print("Saved band comparison to pi_bands_dft_vs_tb_fit.png")

# --- magnetisation: TB vs DFT ---
mag_dft_file = "zgnr_Ny4_M1_mag_AB_strands.npz"
data_mag = np.load(mag_dft_file)

print("DFT magnetisation keys:", data_mag.files)

# ⬇️ adjust these two lines if your keys are named differently
mA_dft = data_mag["mA"]   # DFT magnetisation on A sites (Ny,)
mB_dft = data_mag["mB"]   # DFT magnetisation on B sites (Ny,)

mA_tb = result_tb["mA"]   # TB magnetisation (Ny,)
mB_tb = result_tb["mB"]   # TB magnetisation (Ny,)
strands = np.arange(Ny)

fig2, ax2 = plt.subplots(figsize=(6, 4))

# TB
ax2.plot(strands, mA_tb, marker="o", color="C0", label="TB mA")
ax2.plot(strands, mB_tb, marker="s", color="C1", label="TB mB")

# DFT (dashed)
ax2.plot(strands, mA_dft, "--", marker="o", color="C0", alpha=0.7, label="DFT mA")
ax2.plot(strands, mB_dft, "--", marker="s", color="C1", alpha=0.7, label="DFT mB")

ax2.set_xlabel("strand index m")
ax2.set_ylabel("magnetisation $m_m$")
ax2.set_title(f"Magnetisation profile, Ny={Ny}, t={t_fit:.2f} eV, U={U_fit:.2f} eV")
ax2.legend()
fig2.tight_layout()
fig2.savefig("magnetisation_profile_DFT_vs_TB.png", dpi=300)
print("Saved magnetisation comparison to magnetisation_profile_DFT_vs_TB.png")
