import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = "zgnr_sweep_results"
SUBFOLDER = "central band selection, fitting and band magnetisation graphs"
NPZ_NAME  = "zgnr_pi_smooth_from_Z.npz"

OUT_PNG = "central_pi_bands_gradient.png"

FIGSIZE = (10.5, 6.2)
DPI = 300
LW = 3.0
ALPHA = 0.95

CMAP_NAME = "viridis"   # perceptual, clean
# ============================================================


def ny_key(path):
    base = os.path.basename(path).lower()
    if base.startswith("ny"):
        try:
            return int(base[2:])
        except ValueError:
            return 10**9
    return 10**9


def resolve_sweep_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, BASE_DIR)
    if os.path.isdir(cand):
        return cand

    cwd = os.getcwd()
    cand2 = os.path.join(cwd, BASE_DIR)
    if os.path.isdir(cand2):
        return cand2

    return None


def load_central_bands(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    if "k_dimless" in d:
        k = np.array(d["k_dimless"], float).ravel()
    elif "k" in d:
        k = np.array(d["k"], float).ravel()
    else:
        raise KeyError(f"No k array in {npz_path}")

    if "E_val" not in d or "E_cond" not in d:
        raise KeyError(f"Need E_val and E_cond in {npz_path}")

    return (
        k,
        np.array(d["E_val"], float).ravel(),
        np.array(d["E_cond"], float).ravel(),
    )


def main():
    sweep_dir = resolve_sweep_dir()
    if sweep_dir is None:
        print("[ERROR] Could not locate zgnr_sweep_results.")
        return

    ny_dirs = sorted(
        [p for p in glob.glob(os.path.join(sweep_dir, "Ny*")) if os.path.isdir(p)],
        key=ny_key
    )

    runs = []
    for ny in ny_dirs:
        npz = os.path.join(ny, SUBFOLDER, NPZ_NAME)
        if os.path.isfile(npz):
            runs.append((ny_key(ny), npz))

    if not runs:
        print("[ERROR] No central π-band files found.")
        return

    # ✅ SAFE colormap access for your Matplotlib version
    cmap = plt.get_cmap(CMAP_NAME)
    color_positions = np.linspace(0.15, 0.90, len(runs))

    plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for (Ny, npz), cpos in zip(runs, color_positions):
        k, Eval, Econd = load_central_bands(npz)
        col = cmap(cpos)

        # same color for valence + conduction of same Ny
        ax.plot(k, Econd, color=col, lw=LW, alpha=ALPHA)
        ax.plot(k, Eval,  color=col, lw=LW, alpha=ALPHA)

        ax.plot([], [], color=col, lw=LW, label=rf"$N_y={Ny}$")

    ax.axhline(0.0, color="0.4", lw=1.4, ls="--", zorder=0)

    ax.set_xlabel(r"$ka/\pi$", fontsize=22)
    ax.set_ylabel(r"$E - E_f\ \mathrm{(eV)}$", fontsize=22)

    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=14)

    fig.tight_layout()
    out = os.path.join(sweep_dir, OUT_PNG)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
