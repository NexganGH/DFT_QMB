#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt


def apply_frame_style(ax, frame_lw=1.2, tick_lw=1.0):
    for sp in ax.spines.values():
        sp.set_linewidth(frame_lw)
    ax.tick_params(width=tick_lw, length=6)


def annotate_N_bottom_right(ax, Nlabel: str):
    ax.text(
        0.80, 0.08, rf"$N = {Nlabel}$",   # bottom-right in axes coords
        transform=ax.transAxes,
        fontsize=22,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
    )


def pick_two_from_matrix(M, idx2, name):
    M = np.array(M, float)
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {M.shape}")
    idx2 = np.array(idx2, int).ravel()
    if idx2.size != 2:
        raise ValueError(f"central_indices must have length 2, got {idx2}")
    if np.any(idx2 < 0) or np.any(idx2 >= M.shape[1]):
        raise ValueError(f"{name}: central_indices {idx2} out of bounds for shape {M.shape}")
    return M[:, idx2]


def normalize_01(x):
    x = np.array(x, float).ravel()
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if abs(xmax - xmin) < 1e-15:
        raise ValueError("Cannot normalize x to [0,1]: zero range")
    return (x - xmin) / (xmax - xmin)


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    tb_npz = os.path.join(base, "zgnr6_nonmag_tb_fit_central_only.npz")
    # or:
    # tb_npz = os.path.join(base, "zgnr6_nonmag_tb_fit_full_pi.npz")

    if not os.path.exists(tb_npz):
        raise FileNotFoundError(f"Missing TB fit file: {tb_npz}")

    out_png = os.path.join(base, "zgnr6_central_pi_fit_DFT_vs_TB.png")

    data = np.load(tb_npz, allow_pickle=True)
    keys = list(data.keys())

    required = ["k_dimless", "E_pi_dft", "E_tb", "central_indices"]
    for r in required:
        if r not in data:
            raise KeyError(f"Missing key '{r}' in {tb_npz}. Keys: {keys}")

    x_raw = np.array(data["k_dimless"], float).ravel()
    E_pi_dft = np.array(data["E_pi_dft"], float)   # (nk, nb_pi)
    E_tb = np.array(data["E_tb"], float)           # (nk, nb_pi)
    central_idx = np.array(data["central_indices"], int).ravel()

    nk = x_raw.size
    if E_pi_dft.shape[0] != nk or E_tb.shape[0] != nk:
        raise ValueError(f"Size mismatch: x={nk}, E_pi_dft={E_pi_dft.shape}, E_tb={E_tb.shape}")

    # fix possible 1-based indices
    if central_idx.size != 2:
        raise ValueError(f"central_indices must have 2 elements, got {central_idx}")
    if np.all(central_idx >= 1) and np.max(central_idx) == E_pi_dft.shape[1]:
        central_idx = central_idx - 1

    # pick two central bands
    dft2 = pick_two_from_matrix(E_pi_dft, central_idx, "E_pi_dft")  # (nk,2)
    tb2  = pick_two_from_matrix(E_tb,      central_idx, "E_tb")     # (nk,2)

    # order valence/conduction
    dft_val = np.minimum(dft2[:, 0], dft2[:, 1])
    dft_con = np.maximum(dft2[:, 0], dft2[:, 1])
    tb_val  = np.minimum(tb2[:, 0],  tb2[:, 1])
    tb_con  = np.maximum(tb2[:, 0],  tb2[:, 1])

    # sort by x_raw
    order = np.argsort(x_raw)
    x_raw = x_raw[order]
    dft_val, dft_con = dft_val[order], dft_con[order]
    tb_val, tb_con   = tb_val[order],  tb_con[order]

    # normalize x to [0,1]
    x = normalize_01(x_raw)

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=200)

    # DFT in red (solid)
    ax.plot(x, dft_val, linewidth=3.5, color="#C93B3B", label="DFT")
    ax.plot(x, dft_con, linewidth=3.5, color="#C93B3B")

    # TB in teal (dashed)
    ax.plot(x, tb_val, linewidth=3.0, linestyle="--", color="#137A7F", label="TB")
    ax.plot(x, tb_con, linewidth=3.0, linestyle="--", color="#137A7F")

    ax.axhline(0.0, linestyle="--", linewidth=1.4, color="0.35", alpha=0.8)

    ax.set_xlabel(r"$ka/\pi$", fontsize=18)
    ax.set_ylabel("Energy (eV)", fontsize=18)
    ax.set_xlim(0.0, 1.0)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=14, loc="upper right")

    # N=6 box bottom-right
    annotate_N_bottom_right(ax, "6")

    apply_frame_style(ax, frame_lw=1.2, tick_lw=1.0)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
