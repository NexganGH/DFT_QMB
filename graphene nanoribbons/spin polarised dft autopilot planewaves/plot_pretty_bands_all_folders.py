import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================
ROOT_DIR = "zgnr_sweep_results"   # script sits inside ".../spin polarised dft autopilot planewaves/"
PATTERN  = "*_bands_pi.npz"

EXCLUDE_NY = {1}

OUT_FOLDER   = "pretty_bands"
OUT_COLORED  = "bands_colored.png"   # blue/orange + central red
OUT_GRAY     = "bands_gray.png"      # gray + central red

FIGSIZE = (4, 6.0)  # SAME for both (so you can overlay later)
DPI = 300

# Colors
COL_VAL = "#2E5EAA"   # below Ef  -> blue
COL_CON = "#D89C2B"   # above Ef  -> orange
COL_CEN = "#C23B3B"   # central  -> red
GRAY_OTHER = "0.75"

LW_OTHER   = 1.2
LW_CENTRAL = 3.2
ALPHA_OTHER = 0.95
# ============================================================


def _infer_Ny_from_path(path):
    p = path.replace("\\", "/")
    parts = p.split("/")
    for token in parts[::-1]:
        low = token.lower()
        if low.startswith("ny"):
            try:
                return int(token[2:])
            except Exception:
                pass
    return None


def _to_0_1_axis(k):
    k = np.asarray(k, float).ravel()
    kmin = float(np.min(k))
    kmax = float(np.max(k))
    if np.isclose(kmax, kmin):
        return np.zeros_like(k)
    return (k - kmin) / (kmax - kmin)


def _central_band_curves(Eshift, mu=0.0):
    """
    For each k: pick closest band below mu and closest above mu.
    """
    Nk, nb = Eshift.shape
    val = np.full(Nk, np.nan, float)
    con = np.full(Nk, np.nan, float)

    for i in range(Nk):
        Ek = Eshift[i]
        below = np.where(Ek <= mu)[0]
        above = np.where(Ek >  mu)[0]
        if below.size:
            ib = below[np.argmax(Ek[below])]
            val[i] = Ek[ib]
        if above.size:
            ia = above[np.argmin(Ek[above])]
            con[i] = Ek[ia]
    return val, con


def _band_side_color(y, mu=0.0):
    # majority rule along k
    frac_below = np.mean(y <= mu)
    return COL_VAL if frac_below >= 0.5 else COL_CON


def _plot_common_axes(ax):
    ax.set_xlabel(r"$ka/\pi$")
    ax.set_ylabel(r"$E - E_f$ (eV)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["0", "1"])
    ax.grid(True, alpha=0.25)


def _annotate_N(ax, Ny):
    ax.text(
        0.97, 0.93, f"$N={Ny}$",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
        zorder=10,
    )


def _load_k_and_E(npz_path):
    """
    Your files contain:
      k_frac, k_dimless, k_plot
      energies, E_rel, efermi
      bands_up, bands_dn

    Strategy:
      - Use k_plot if present, else k_dimless, else k_frac.
      - Prefer E_rel if present (already E - Ef), otherwise use 'energies' and subtract efermi.
      - Ensure E is (Nk, nbands) by averaging spins if needed.
    """
    d = np.load(npz_path, allow_pickle=True)

    # ---- k ----
    if "k_plot" in d:
        k = np.asarray(d["k_plot"], float).ravel()
    elif "k_dimless" in d:
        k = np.asarray(d["k_dimless"], float).ravel()
    elif "k_frac" in d:
        k = np.asarray(d["k_frac"], float).ravel()
    else:
        raise KeyError(f"[{npz_path}] Cannot find k array. Keys: {list(d.keys())}")

    Nk = len(k)

    # ---- energies ----
    if "E_rel" in d:
        Eraw = np.asarray(d["E_rel"], float)
        # E_rel is expected already relative to Ef
        mu = 0.0
    elif "energies" in d:
        Eraw = np.asarray(d["energies"], float)
        mu = float(np.array(d["efermi"]).item()) if "efermi" in d else 0.0
    elif ("bands_up" in d) and ("bands_dn" in d):
        Eup = np.asarray(d["bands_up"], float)
        Edn = np.asarray(d["bands_dn"], float)
        # handle shapes later
        Eraw = 0.5 * (Eup + Edn)
        mu = float(np.array(d["efermi"]).item()) if "efermi" in d else 0.0
    else:
        raise KeyError(f"[{npz_path}] Cannot find energies. Keys: {list(d.keys())}")

    # ---- shape normalization to (Nk, nbands) ----
    if Eraw.ndim == 2:
        # could be (Nk, nb) OR (nb, Nk)
        if Eraw.shape[0] == Nk:
            E = Eraw
        elif Eraw.shape[1] == Nk:
            E = Eraw.T
        else:
            raise ValueError(f"[{npz_path}] E shape {Eraw.shape} incompatible with Nk={Nk}")
    elif Eraw.ndim == 3:
        # could be (nspins, Nk, nb) or (Nk, nb, nspins)
        if Eraw.shape[1] == Nk:
            # (nspins, Nk, nb)
            E = np.mean(Eraw, axis=0)
        elif Eraw.shape[0] == Nk:
            # (Nk, nb, nspins)
            E = np.mean(Eraw, axis=-1)
        else:
            raise ValueError(f"[{npz_path}] E shape {Eraw.shape} incompatible with Nk={Nk}")
    else:
        raise ValueError(f"[{npz_path}] Unrecognized E ndim={Eraw.ndim}")

    # Shift to Ef = 0 (unless already E_rel)
    Eshift = E - mu
    return k, Eshift


def make_two_plots(npz_path):
    case_dir = os.path.dirname(npz_path)
    out_dir = os.path.join(case_dir, OUT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)

    Ny = _infer_Ny_from_path(npz_path)

    k, Eshift = _load_k_and_E(npz_path)
    x = _to_0_1_axis(k)

    # central (closest to 0 from below/above)
    val_c, con_c = _central_band_curves(Eshift, mu=0.0)

    # ------------------- COLORED -------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for b in range(Eshift.shape[1]):
        y = Eshift[:, b]
        c = _band_side_color(y, mu=0.0)
        ax.plot(x, y, lw=LW_OTHER, alpha=ALPHA_OTHER, color=c, zorder=1)

    ax.plot(x, val_c, lw=LW_CENTRAL, color=COL_CEN, zorder=3)
    ax.plot(x, con_c, lw=LW_CENTRAL, color=COL_CEN, zorder=3)

    ax.axhline(0.0, ls="--", lw=1.2, alpha=0.7, zorder=0)

    _plot_common_axes(ax)
    if Ny is not None:
        _annotate_N(ax, Ny)

    fig.tight_layout()
    out1 = os.path.join(out_dir, OUT_COLORED)
    fig.savefig(out1, dpi=DPI)
    plt.close(fig)

    # ------------------- GRAY -------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for b in range(Eshift.shape[1]):
        y = Eshift[:, b]
        ax.plot(x, y, lw=LW_OTHER, alpha=1.0, color=GRAY_OTHER, zorder=1)

    ax.plot(x, val_c, lw=LW_CENTRAL, color=COL_CEN, zorder=3)
    ax.plot(x, con_c, lw=LW_CENTRAL, color=COL_CEN, zorder=3)

    ax.axhline(0.0, ls="--", lw=1.2, alpha=0.7, zorder=0)

    _plot_common_axes(ax)
    if Ny is not None:
        _annotate_N(ax, Ny)

    fig.tight_layout()
    out2 = os.path.join(out_dir, OUT_GRAY)
    fig.savefig(out2, dpi=DPI)
    plt.close(fig)

    return out1, out2


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(this_dir, ROOT_DIR))

    if not os.path.isdir(root):
        print(f"[ERROR] ROOT_DIR not found:\n  {root}")
        return

    files = sorted(glob.glob(os.path.join(root, "**", PATTERN), recursive=True))
    if not files:
        print(f"[ERROR] No files matching {PATTERN} under:\n  {root}")
        return

    kept = []
    for f in files:
        Ny = _infer_Ny_from_path(f)
        if Ny in EXCLUDE_NY:
            continue
        kept.append(f)

    if not kept:
        print("[ERROR] After excluding Ny1, no band files remain.")
        return

    print(f"[INFO] Found {len(files)} band files, using {len(kept)} after excluding Ny1.")

    for f in kept:
        try:
            out1, out2 = make_two_plots(f)
            print("Saved:", out1)
            print("Saved:", out2)
        except Exception as e:
            print(f"[WARN] Skipping {f}\n   -> {e}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
