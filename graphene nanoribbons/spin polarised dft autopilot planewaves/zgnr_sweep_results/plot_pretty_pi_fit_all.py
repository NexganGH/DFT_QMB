# plot_pretty_pi_fit_all.py
#
# Places the info box (N, t, U) inside the axes, automatically choosing
# between two user-preferred "empty spots" (left-middle or bottom-right),
# selecting the one with LESS overlap with the curves.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================
ROOT = "."
SUBFOLDER = "central band selection, fitting and band magnetisation graphs"

DFT_NPZ = "zgnr_pi_smooth_from_Z.npz"
TB_NPZ_PATTERN = "zgnr_*_tb_fit.npz"

OUT_DIRNAME = "pi_bands_pretty"
OUT_NAME = "pi_bands_pretty.png"

FIGSIZE = (7.6, 4.4)
DPI = 320

COLOR_DFT = "tab:red"
COLOR_TB  = "tab:blue"
LS_DFT = "-"
LS_TB  = "--"

LW_DFT = 3.2
LW_TB  = 2.8

FONTSIZE = 13
LEG_FONTSIZE = 12
BOX_FONTSIZE = 12

# Two preferred spots (axes fraction coordinates)
# 1) Left-middle-ish empty area you marked
SPOT_LEFT  = dict(x=0.18, y=0.52, ha="center", va="center")
# 2) Bottom-right empty area you marked
SPOT_RIGHT = dict(x=0.84, y=0.24, ha="center", va="center")

# How strict overlap scoring is (points are sampled along each curve)
N_SAMPLE_POINTS = 500


# =========================
# Helpers
# =========================
def safe_get(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_dft_central(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    k  = safe_get(d, ["k_dimless", "k_norm", "k", "kpath", "x"])
    Ev = safe_get(d, ["E_val", "E_v", "Eval", "E_valence"])
    Ec = safe_get(d, ["E_cond", "E_c", "Econd", "E_conduction"])

    if k is None:
        raise KeyError(f"No k array found. Keys: {list(d.keys())}")
    if Ev is None or Ec is None:
        raise KeyError(f"Missing E_val/E_cond. Keys: {list(d.keys())}")

    return np.array(k, float).ravel(), np.array(Ev, float).ravel(), np.array(Ec, float).ravel()


def load_tb_fit(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    k  = safe_get(d, ["k_dimless", "k_norm", "k", "kpath", "x"])
    Ev = safe_get(d, ["E_val_tb", "E_tb_val", "E_val_fit", "E_v_fit", "E_val"])
    Ec = safe_get(d, ["E_cond_tb", "E_tb_cond", "E_cond_fit", "E_c_fit", "E_cond"])

    t  = safe_get(d, ["t_fit", "t", "tbest", "t_opt"])
    U  = safe_get(d, ["U_fit", "U", "Ubest", "U_opt"])
    Ny = safe_get(d, ["Ny", "N"])

    if k is None or Ev is None or Ec is None:
        raise KeyError(f"Missing k/E_val_tb/E_cond_tb. Keys: {list(d.keys())}")

    t_val  = float(np.array(t).item()) if t is not None else None
    U_val  = float(np.array(U).item()) if U is not None else None
    Ny_val = int(np.array(Ny).item()) if Ny is not None else None

    return np.array(k, float).ravel(), np.array(Ev, float).ravel(), np.array(Ec, float).ravel(), t_val, U_val, Ny_val


def _interp_resample(x, y, n=N_SAMPLE_POINTS):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xs = np.linspace(x.min(), x.max(), n)
    ys = np.interp(xs, x, y)
    return xs, ys


def _score_overlap(ax, renderer, bbox_display, curve_xy_list):
    """
    bbox_display: Bbox in DISPLAY coords
    curve_xy_list: list of (xdata, ydata) arrays in DATA coords
    Returns: overlap score = number of sampled curve points inside bbox_display
    """
    score = 0
    for (xdata, ydata) in curve_xy_list:
        xs, ys = _interp_resample(xdata, ydata, n=N_SAMPLE_POINTS)
        # Convert curve points to display coords
        pts_disp = ax.transData.transform(np.column_stack([xs, ys]))
        x_disp = pts_disp[:, 0]
        y_disp = pts_disp[:, 1]
        inside = (
            (x_disp >= bbox_display.x0) & (x_disp <= bbox_display.x1) &
            (y_disp >= bbox_display.y0) & (y_disp <= bbox_display.y1)
        )
        score += int(np.count_nonzero(inside))
    return score


def place_info_box(ax, fig, info_text, spots, curve_xy_list):
    """
    Try each spot, compute overlap score, pick the best.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    best = None
    best_score = None
    best_artist = None

    # Try in given order, but choose minimal overlap
    for sp in spots:
        # Create candidate text
        txt = ax.text(
            sp["x"], sp["y"], info_text,
            transform=ax.transAxes,
            ha=sp["ha"], va=sp["va"],
            fontsize=BOX_FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95),
            zorder=10
        )
        fig.canvas.draw()
        bbox = txt.get_window_extent(renderer=renderer)

        score = _score_overlap(ax, renderer, bbox, curve_xy_list)

        # Keep the best
        if best_score is None or score < best_score:
            # remove previous best artist (if any) later; for now store
            if best_artist is not None:
                best_artist.remove()
            best_artist = txt
            best_score = score
            best = sp
        else:
            # not best, remove this candidate
            txt.remove()

    # best_artist remains on axes
    return best, best_score


def plot_pretty(k_dft, Ev_dft, Ec_dft, k_tb, Ev_tb, Ec_tb, t, U, case_N, outpath):
    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axhline(0.0, lw=1.2, ls="--", color="0.55", zorder=0)

    # Plot and keep data references for overlap scoring
    ax.plot(k_dft, Ev_dft, color=COLOR_DFT, lw=LW_DFT, ls=LS_DFT, label="DFT (central π)")
    ax.plot(k_dft, Ec_dft, color=COLOR_DFT, lw=LW_DFT, ls=LS_DFT, label="_nolegend_")

    ax.plot(k_tb, Ev_tb, color=COLOR_TB, lw=LW_TB, ls=LS_TB, label="TB fit")
    ax.plot(k_tb, Ec_tb, color=COLOR_TB, lw=LW_TB, ls=LS_TB, label="_nolegend_")

    ax.set_xlabel(r"$ka/\pi$")
    ax.set_ylabel(r"$E - E_f$ (eV)")

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=LEG_FONTSIZE)

    # Build info text
    case_text = f"N = {case_N}" if case_N is not None else "N = ?"
    t_str = f"{t:.4f}" if t is not None else "—"
    U_str = f"{U:.4f}" if U is not None else "—"
    info_text = rf"${case_text}$" + "\n" + rf"$t = {t_str}\ \mathrm{{eV}}$" + "\n" + rf"$U = {U_str}\ \mathrm{{eV}}$"

    # Candidate spots: you want one of these two
    spots = [SPOT_RIGHT, SPOT_LEFT]  # prefer right, fallback left if needed

    curves = [
        (k_dft, Ev_dft),
        (k_dft, Ec_dft),
        (k_tb, Ev_tb),
        (k_tb, Ec_tb),
    ]

    place_info_box(ax, fig, info_text, spots, curves)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def main():
    ny_dirs = [p for p in glob.glob(os.path.join(ROOT, "Ny*")) if os.path.isdir(p)]
    ny_dirs = sorted(
        ny_dirs,
        key=lambda p: int(os.path.basename(p)[2:]) if os.path.basename(p)[2:].isdigit() else 10**9
    )

    if not ny_dirs:
        print("[ERROR] No Ny* folders found in:", os.path.abspath(ROOT))
        return

    ok = 0
    fail = 0

    for ny in ny_dirs:
        work = os.path.join(ny, SUBFOLDER)
        if not os.path.isdir(work):
            print(f"[SKIP] {ny}: missing subfolder '{SUBFOLDER}'")
            fail += 1
            continue

        dft_path = os.path.join(work, DFT_NPZ)
        if not os.path.isfile(dft_path):
            print(f"[SKIP] {ny}: missing {DFT_NPZ}")
            fail += 1
            continue

        tb_candidates = sorted(glob.glob(os.path.join(work, TB_NPZ_PATTERN)))
        if not tb_candidates:
            print(f"[SKIP] {ny}: missing TB fit npz ({TB_NPZ_PATTERN})")
            fail += 1
            continue
        tb_path = tb_candidates[0]

        try:
            k_dft, Ev_dft, Ec_dft = load_dft_central(dft_path)
            k_tb, Ev_tb, Ec_tb, t, U, Ny_from_tb = load_tb_fit(tb_path)
        except Exception as e:
            print(f"[FAIL] {ny}: {e}")
            fail += 1
            continue

        case_N = Ny_from_tb
        if case_N is None:
            base = os.path.basename(ny)
            if base.lower().startswith("ny"):
                try:
                    case_N = int(base[2:])
                except Exception:
                    case_N = None

        out_dir = os.path.join(work, OUT_DIRNAME)
        outpath = os.path.join(out_dir, OUT_NAME)

        plot_pretty(k_dft, Ev_dft, Ec_dft, k_tb, Ev_tb, Ec_tb, t, U, case_N, outpath)
        print(f"[OK] {ny}: saved -> {outpath}")
        ok += 1

    print("\nDONE.")
    print(f"  OK   : {ok}")
    print(f"  FAIL : {fail}")


if __name__ == "__main__":
    main()
