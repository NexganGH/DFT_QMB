#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gpaw import restart


# ----------------------------
# File discovery
# ----------------------------
def find_gpw(run_dir: str) -> str | None:
    files = os.listdir(run_dir)
    pref = [f for f in files if f.endswith("_relaxed.gpw")]
    if pref:
        return os.path.join(run_dir, sorted(pref)[0])
    anyg = [f for f in files if f.endswith(".gpw")]
    if anyg:
        return os.path.join(run_dir, sorted(anyg)[0])
    return None


# ----------------------------
# Plot styling
# ----------------------------
def apply_frame_style(ax, frame_lw=1.2, tick_lw=1.0):
    for sp in ax.spines.values():
        sp.set_linewidth(frame_lw)
    ax.tick_params(width=tick_lw, length=6)


def annotate_N(ax, Nlabel: str):
    ax.text(
        0.70, 0.92, rf"$N = {Nlabel}$",
        transform=ax.transAxes,
        fontsize=22,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
    )


# ----------------------------
# Central band tracking
# ----------------------------
def track_two_bands_smooth(x, E, Ef=0.0, alpha=0.35):
    """
    Track 2 bands closest to Ef with smooth continuation across crossings.

    Inputs:
      x: (nk,)
      E: (nk, nb) absolute energies (eV)
      Ef: float (eV)
    Returns:
      idx: (nk, 2) indices of tracked bands
      Etrk: (nk, 2) absolute energies for tracked bands
    """
    nk, nb = E.shape
    y = E - Ef

    idx = np.full((nk, 2), -1, dtype=int)
    Etrk = np.full((nk, 2), np.nan, dtype=float)

    # init: pick two closest to 0 at x[0]
    d0 = np.abs(y[0, :])
    pick = np.argsort(d0)[:2]
    pick = pick[np.argsort(y[0, pick])]  # lower then upper
    idx[0] = pick
    Etrk[0] = E[0, pick]

    # slope init in y-space
    dE_prev = np.zeros(2, float)
    if nk > 1:
        dx = (x[1] - x[0]) + 1e-15
        dE_prev = (y[1, pick] - y[0, pick]) / dx

    for i in range(1, nk):
        dx = (x[i] - x[i - 1]) + 1e-15
        prev_idx = idx[i - 1]
        prev_y = y[i - 1, prev_idx]
        prev_E = E[i - 1, prev_idx]

        costs = np.zeros((2, nb), float)
        for j in range(2):
            dy_now = (y[i, :] - prev_y[j]) / dx
            costs[j, :] = np.abs(y[i, :] - prev_y[j]) + alpha * np.abs(dy_now - dE_prev[j])

        c0 = np.argsort(costs[0])[:12]
        c1 = np.argsort(costs[1])[:12]
        best_pair, best_val = None, np.inf
        for a in c0:
            for b in c1:
                if a == b:
                    continue
                v = costs[0, a] + costs[1, b]
                if v < best_val:
                    best_val, best_pair = v, (a, b)

        if best_pair is None:
            a = int(np.argmin(costs[0]))
            b = int(np.argmin(costs[1]))
            if a == b:
                b = int(np.argsort(costs[1])[1])
            best_pair = (a, b)

        idx[i] = best_pair
        Etrk[i, 0] = E[i, best_pair[0]]
        Etrk[i, 1] = E[i, best_pair[1]]

        # update slope in y-space
        dE_prev[0] = ((Etrk[i, 0] - Ef) - (prev_E[0] - Ef)) / dx
        dE_prev[1] = ((Etrk[i, 1] - Ef) - (prev_E[1] - Ef)) / dx

        # keep lower/upper ordering
        if (Etrk[i, 0] - Ef) > (Etrk[i, 1] - Ef):
            Etrk[i] = Etrk[i, ::-1]
            idx[i] = idx[i, ::-1]
            dE_prev[:] = dE_prev[::-1]

    return idx, Etrk


# ----------------------------
# Extract from .gpw
# ----------------------------
def extract_bands_from_gpw(gpw_path: str):
    """
    Returns:
      kpts_red: (nk,3) reduced coordinates (IBZ)
      x: (nk,) mapped to [0,1] (ka/pi axis)
      E: (nk, nbands) eigenvalues (eV)
      Ef: float (eV)
    """
    atoms, calc = restart(gpw_path, txt=None)

    Ef = float(calc.get_fermi_level())

    # k-points in reduced coordinates
    kpts = np.array(calc.get_ibz_k_points(), float)
    nk = len(kpts)
    nb = int(calc.get_number_of_bands())

    # eigenvalues in eV
    E = np.zeros((nk, nb), float)
    for i in range(nk):
        E[i, :] = calc.get_eigenvalues(kpt=i)

    # Detect which reduced component varies the most (robust)
    # For your case it will pick z automatically.
    spans = np.ptp(kpts, axis=0)  # max-min for each column
    idir = int(np.argmax(spans))  # 0,1,2

    k_disp = kpts[:, idir].copy()

    # IMPORTANT: normalize exactly once to [0,1]
    denom = (k_disp.max() - k_disp.min())
    if denom < 1e-14:
        raise ValueError(f"Dispersion direction has zero span in {gpw_path}. kpts span={spans}")

    x = (k_disp - k_disp.min()) / denom  # -> [0,1]

    # Sort by x
    order = np.argsort(x)
    x = x[order]
    kpts = kpts[order]
    E = E[order]

    return kpts, x, E, Ef, idir


# ----------------------------
# Plotting
# ----------------------------
def plot_full_colored(x, E, Ef, idx_tracked, outpath,
                      frame_lw=1.2, tick_lw=1.0, ylims=None, Nlabel=""):
    fig, ax = plt.subplots(figsize=(6, 10), dpi=220)

    nk, nb = E.shape
    y = E - Ef

    # Plot all bands: orange above, blue below
    for b in range(nb):
        yy = y[:, b]
        if np.nanmean(yy) >= 0:
            ax.plot(x, yy, linewidth=1.2, color="#D99A1C")
        else:
            ax.plot(x, yy, linewidth=1.2, color="#2D62B8")

    # Highlight tracked central bands
    if idx_tracked is not None:
        for j in range(2):
            ax.plot(x, y[np.arange(nk), idx_tracked[:, j]], color="#C93B3B", linewidth=3.5)

    # Ef line
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="#6BA3D6")

    ax.set_xlabel(r"$ka/\pi$", fontsize=18)
    ax.set_ylabel(r"$E - E_f\ \mathrm{(eV)}$", fontsize=18)

    # FORCE x-range exactly [0,1]
    ax.set_xlim(0.0, 1.0)

    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.grid(True, alpha=0.25)
    if Nlabel:
        annotate_N(ax, Nlabel)

    apply_frame_style(ax, frame_lw=frame_lw, tick_lw=tick_lw)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_central_highlight(x, E, Ef, idx_tracked, outpath,
                          frame_lw=1.2, tick_lw=1.0, ylims=None, Nlabel="",
                          gray_alpha=0.35):
    if idx_tracked is None:
        return

    fig, ax = plt.subplots(figsize=(6, 10), dpi=220)

    nk, nb = E.shape
    y = E - Ef
    central = set(idx_tracked.ravel().tolist())

    for b in range(nb):
        if b in central:
            ax.plot(x, y[:, b], color="#C93B3B", linewidth=3.5)
        else:
            ax.plot(x, y[:, b], color="0.7", linewidth=1.3, alpha=gray_alpha)

    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="#6BA3D6")

    ax.set_xlabel(r"$ka/\pi$", fontsize=18)
    ax.set_ylabel(r"$E - E_f\ \mathrm{(eV)}$", fontsize=18)

    # FORCE x-range exactly [0,1]
    ax.set_xlim(0.0, 1.0)

    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.grid(True, alpha=0.25)
    if Nlabel:
        annotate_N(ax, Nlabel)

    apply_frame_style(ax, frame_lw=frame_lw, tick_lw=tick_lw)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=".", help="Path to runs directory (default: current)")
    ap.add_argument("--outfolder", default="central_bands_from_gpw", help="Output folder inside each ZGNR-*")
    ap.add_argument("--alpha", type=float, default=0.35, help="Smooth tracking weight (default 0.35)")
    ap.add_argument("--frame", type=float, default=1.2, help="Frame linewidth")
    ap.add_argument("--ticks", type=float, default=1.0, help="Tick linewidth")
    ap.add_argument("--ymin", type=float, default=-21.0)
    ap.add_argument("--ymax", type=float, default=11.0)
    args = ap.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    ylims = (args.ymin, args.ymax)

    folders = sorted([
        d for d in os.listdir(runs_dir)
        if d.startswith("ZGNR-") and os.path.isdir(os.path.join(runs_dir, d))
    ])
    if not folders:
        raise RuntimeError(f"No ZGNR-* folders found in {runs_dir}")

    print(f"[INFO] Found folders: {folders}")

    for d in folders:
        run_dir = os.path.join(runs_dir, d)
        gpw = find_gpw(run_dir)
        if gpw is None:
            print(f"[SKIP] {d}: no .gpw found")
            continue

        print(f"\n[RUN] {d}")
        print(f"      gpw: {os.path.basename(gpw)}")

        kpts, x, E, Ef, idir = extract_bands_from_gpw(gpw)

        # Track central bands relative to Ef
        idx, Etrk = track_two_bands_smooth(x, E, Ef=Ef, alpha=args.alpha)

        outdir = os.path.join(run_dir, args.outfolder)
        os.makedirs(outdir, exist_ok=True)

        out_npz = os.path.join(outdir, "full_bands_from_gpw.npz")
        np.savez(
            out_npz,
            kpts_reduced=kpts,
            x_ka_over_pi=x,     # guaranteed in [0,1]
            E_full=E,
            Ef=Ef,
            idx_tracked=idx,
            E_tracked=Etrk,
            source_gpw=gpw,
            alpha=args.alpha,
            dispersion_component=idir,  # 0,1,2 (which reduced coordinate varied most)
        )
        print(f"      saved: {out_npz}")
        print(f"      Ef: {Ef:.6f} eV, E shape: {E.shape}, x range: [{x.min():.3f}, {x.max():.3f}]")

        # label box uses N from folder name
        Nlabel = d.replace("ZGNR-", "").strip()

        out_full = os.path.join(outdir, "bands_full_colored.png")
        out_cent = os.path.join(outdir, "bands_central_highlight.png")

        plot_full_colored(
            x, E, Ef, idx, out_full,
            frame_lw=args.frame, tick_lw=args.ticks,
            ylims=ylims, Nlabel=Nlabel
        )
        plot_central_highlight(
            x, E, Ef, idx, out_cent,
            frame_lw=args.frame, tick_lw=args.ticks,
            ylims=ylims, Nlabel=Nlabel
        )

        print(f"      plot : {out_full}")
        print(f"      plot : {out_cent}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
