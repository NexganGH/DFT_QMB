#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autopilot for ZGNR widths: runs DFT relax -> bands/DOS -> π-band selection
-> TB fit (full π and central-only) -> TB DOS.
Saves all arrays as .npz and both PNG and zoomable HTML plots.

Requires: ase, gpaw, numpy, scipy, matplotlib, tqdm
Optional (for interactive/zoomable plots): plotly

Folder layout (expected):
- this script
- config_zgnr.py    (as provided)
- 01/02/03/04 scripts not used directly (we re-implement with params)
- TB code: config_zgnr.get_tb_module() -> module with H_zgnr_k(k, N, t, a)

Outputs per width in: ./runs/ZGNR-{N}/
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm

# plotting (PNG always; HTML if plotly is installed)
import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# DFT / ASE / GPAW
from ase.build import graphene_nanoribbon
from ase.optimize import BFGS
from ase.io import write
from gpaw import GPAW, PW

from scipy.optimize import minimize

import sys
sys.path.append("/mnt/c/Users/feder/PycharmProjects/DFT_QMB/graphene nanoribbons/normal dft saturated")
import config_zgnr as cfg



# -----------------------------
# ---- Master configuration ---
# -----------------------------
@dataclass
class SweepConfig:
    widths: tuple = (1, 2, 4, 6, 8)
    length_repeats: int = cfg.LENGTH_REPEATS
    vacuum: float = cfg.VACUUM
    # DFT
    ecut: float = cfg.ECUT
    fmax: float = cfg.FMAX
    kpts_relax: int = cfg.KPTS_1D_RELAX
    nk_path: int = cfg.NK_PATH
    nbands: int = cfg.NBANDS
    eta_dos: float = cfg.ETA_DOS
    # plotting / IO
    show_plots: bool = False        # set True to plt.show() while running
    out_root: str = "runs"
    # TB DOS resolution (denser than DFT path for smooth DOS)
    nk_tb_dos: int = 2000
    # Nelder–Mead options
    nm_maxiter_full: int = 600
    nm_maxiter_central: int = 500


# ----------------------------------------
# ----- Helpers: plotting (PNG/HTML) -----
# ----------------------------------------
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def fig_save_png(fig, path_png):
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    plt.close(fig)


def plot_bands_png(k_dimless, bands_list, labels, title, out_png, ef_line=True):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    x = k_dimless / np.pi
    for E, style, lw, color, lab in bands_list:
        for n in range(E.shape[1]):
            ax.plot(x, E[:, n], style, lw=lw, color=color, alpha=0.95 if lw > 1 else 0.8, label=lab if n == 0 else None)
    if ef_line:
        ax.axhline(0.0, ls="--", lw=0.7, color="k", alpha=0.6)
    ax.set_xlabel(r"$k a / \pi$")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)
    if any(lab for *_r, lab in bands_list):
        ax.legend(loc="best", fontsize=9)
    fig_save_png(fig, out_png)


def plot_dos_png(E_grid, curves, title, out_png, ef_line=True):
    fig, ax = plt.subplots(figsize=(4.6, 5))
    for (y, label, style, lw) in curves:
        ax.plot(y, E_grid, style, lw=lw, label=label)
    if ef_line:
        ax.axhline(0.0, ls="--", lw=0.7, color="k", alpha=0.6)
    ax.set_xlabel("DOS (states/eV/cell)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig_save_png(fig, out_png)


def plot_bands_html(k_dimless, datasets, title, out_html):
    if not _HAS_PLOTLY:
        return
    fig = go.Figure()
    x = (k_dimless / np.pi).tolist()
    for name, E, dash in datasets:
        for n in range(E.shape[1]):
            fig.add_trace(go.Scatter(
                x=x, y=E[:, n].tolist(),
                mode="lines",
                name=f"{name}" if n == 0 else f"{name} (cont.)",
                line=dict(dash=dash)
            ))
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig.update_layout(
        title=title, xaxis_title="k a / π", yaxis_title="Energy (eV)",
        legend=dict(itemsizing="constant")
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_dos_html(E_grid, curves, title, out_html):
    if not _HAS_PLOTLY:
        return
    fig = go.Figure()
    for (y, label) in curves:
        fig.add_trace(go.Scatter(x=y.tolist(), y=E_grid.tolist(), mode="lines", name=label))
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig.update_layout(title=title, xaxis_title="DOS (states/eV/cell)", yaxis_title="Energy (eV)")
    fig.write_html(out_html, include_plotlyjs="cdn")


# ----------------------------------------
# --------- DFT stage functions ----------
# ----------------------------------------
def make_zgnr(N_zigzag: int, length_repeats: int, vacuum: float):
    """
    Non-magnetic ZGNR with hydrogen-saturated edges (saturated=True).
    NOTE: Your original file had a comment saying "non-saturated", but the code
          used saturated=True. Here we keep saturated=True to match the code.
    """
    atoms = graphene_nanoribbon(
        N_zigzag,
        length_repeats,
        type="zigzag",
        saturated=True,   # keep as in your code
        vacuum=vacuum,
        magnetic=False,
    )
    atoms.pbc = (True, True, True)
    atoms.center()  # keep unit cell, shift coordinates to center
    # keep a small margin from boundaries in scaled coords to avoid GPAW warnings
    spos = atoms.get_scaled_positions()
    eps = 0.05
    spos = np.clip(spos, eps, 1.0 - eps)
    atoms.set_scaled_positions(spos)
    return atoms


def relax_zgnr_to_gpw(N: int, out_dir: str, sc: SweepConfig):
    label = f"zgnr{N}_nonmag"
    gpw_file = os.path.join(out_dir, f"{label}_relaxed.gpw")

    atoms = make_zgnr(N, sc.length_repeats, sc.vacuum)
    calc = GPAW(
        mode=PW(sc.ecut),
        xc="PBE",
        kpts=(1, 1, sc.kpts_relax),
        spinpol=False,
        txt=os.path.join(out_dir, f"{label}_relax.txt"),
    )
    atoms.calc = calc
    dyn = BFGS(
        atoms,
        trajectory=os.path.join(out_dir, f"{label}_relax.traj"),
        logfile=os.path.join(out_dir, f"{label}_relax.log"),
    )
    dyn.run(fmax=sc.fmax)

    write(os.path.join(out_dir, f"{label}_relaxed.xyz"), atoms)
    calc.write(gpw_file, mode="all")
    return gpw_file, label


def bands_and_dos_from_gpw(gpw_file: str, label: str, out_dir: str, sc: SweepConfig):
    calc = GPAW(gpw_file)
    atoms = calc.get_atoms()

    # periodic direction = x
    a_dft = atoms.cell.lengths()[0]

    # band path Γ -> X along x
    path = atoms.cell.bandpath("GX", npoints=sc.nk_path)
    kpts = path.kpts
    nk = len(kpts)

    bs_calc = calc.fixed_density(
        kpts=kpts,
        symmetry="off",
        nbands=sc.nbands,
        txt=os.path.join(out_dir, f"{label}_bands.txt"),
    )

    ef = bs_calc.get_fermi_level()
    E_all = np.zeros((nk, sc.nbands), dtype=float)
    for ik in range(nk):
        eigs = bs_calc.get_eigenvalues(kpt=ik)
        E_all[ik, :min(len(eigs), sc.nbands)] = eigs[:sc.nbands]
    E_rel_all = E_all - ef

    k_dimless = np.linspace(0.0, np.pi, nk)

    # DOS with Gaussian broadening
    E_flat = E_rel_all.ravel()
    Emin, Emax = E_flat.min(), E_flat.max()
    pad = 0.2 * (Emax - Emin) if (Emax > Emin) else 1.0
    Emin -= pad
    Emax += pad
    n_E = 2000
    E_grid = np.linspace(Emin, Emax, n_E)
    x = E_grid[:, None] - E_flat[None, :]
    gaussians = np.exp(-0.5 * (x / sc.eta_dos) ** 2) / (np.sqrt(2 * np.pi) * sc.eta_dos)
    DOS = gaussians.sum(axis=1) / nk

    # save arrays
    np.savez(
        os.path.join(out_dir, f"{label}_bands_dos.npz"),
        k_dimless=k_dimless,
        E_rel_all=E_rel_all,
        E_grid=E_grid,
        DOS=DOS,
        a_dft=a_dft,
        ef=ef,
    )

    # plots
    plot_bands_png(
        k_dimless,
        bands_list=[(E_rel_all, "-", 0.6, "C0", "DFT bands")],
        labels=None,
        title=f"DFT bands (non-magnetic) – {label}",
        out_png=os.path.join(out_dir, f"{label}_bands.png"),
    )
    if _HAS_PLOTLY:
        plot_bands_html(
            k_dimless,
            datasets=[("DFT", E_rel_all, "solid")],
            title=f"DFT bands (non-magnetic) – {label}",
            out_html=os.path.join(out_dir, f"{label}_bands.html"),
        )

    plot_dos_png(
        E_grid,
        curves=[(DOS, "DFT DOS", "-", 1.2)],
        title=f"DFT DOS – {label}",
        out_png=os.path.join(out_dir, f"{label}_dos.png"),
    )
    if _HAS_PLOTLY:
        plot_dos_html(
            E_grid,
            curves=[(DOS, "DFT DOS")],
            title=f"DFT DOS – {label}",
            out_html=os.path.join(out_dir, f"{label}_dos.html"),
        )

    if sc.show_plots:
        plt.show()

    return k_dimless, E_rel_all, E_grid, DOS, a_dft


# ----------------------------------------
# -------- π selection & TB fitting ------
# ----------------------------------------
def select_pi_bands(E_rel_all: np.ndarray, N_zigzag: int):
    """
    Select 2*N_zigzag π-bands closest to EF by smallest avg |E|.
    Returns: E_pi_sorted (Nk, 2N), band_indices (2N,)
    """
    Nk, nbands = E_rel_all.shape
    n_pi = 2 * N_zigzag
    if nbands < n_pi:
        raise ValueError(f"Not enough bands ({nbands}) to extract 2*N={n_pi} π-bands.")
    avg_abs = np.mean(np.abs(E_rel_all), axis=0)
    band_indices = np.sort(np.argsort(avg_abs)[:n_pi])
    E_pi = E_rel_all[:, band_indices]
    E_pi_sorted = np.sort(E_pi, axis=1)
    return E_pi_sorted, band_indices


def build_central_indices(n_pi: int):
    """Return indices [i_valence_top, i_conduction_bottom] for central pair."""
    i1 = n_pi // 2 - 1
    i2 = n_pi // 2
    return [i1, i2]


def tb_bands_on_kgrid(k_dimless: np.ndarray, N: int, t: float, a: float, Hk_func):
    Nk = len(k_dimless)
    dim = 2 * N
    E_tb = np.zeros((Nk, dim), dtype=float)
    ks = k_dimless / a
    for i, k in enumerate(ks):
        Hk = Hk_func(k, N, t=t, a=a)
        w, _ = np.linalg.eigh(Hk)
        E_tb[i, :] = np.sort(w.real)
    return E_tb


def fit_tb_full_pi(k_dimless, E_pi_dft, N, a, Hk_func, x0=(-2.7, 0.0), maxiter=600):
    """
    Fit t and global shift dE to all π bands (2N).
    """
    def loss(params):
        t, dE = params
        E_tb = tb_bands_on_kgrid(k_dimless, N, t, a, Hk_func) + dE
        return np.mean((E_tb - E_pi_dft) ** 2)

    res = minimize(loss, np.array(x0), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4})
    t_opt, dE_opt = res.x
    E_tb_opt = tb_bands_on_kgrid(k_dimless, N, t_opt, a, Hk_func) + dE_opt
    rmse = np.sqrt(np.mean((E_tb_opt - E_pi_dft) ** 2))
    return dict(t=t_opt, dE=dE_opt, rmse=rmse, E_tb=E_tb_opt, success=res.success, message=res.message)


def fit_tb_central_only(k_dimless, E_pi_dft, N, a, Hk_func, maxiter=500, t0=-2.7):
    """
    Fit only t (negative enforced) to the two central π bands near EF.
    """
    n_pi = E_pi_dft.shape[1]
    cidx = build_central_indices(n_pi)

    def loss(x):
        t = -abs(x[0])
        E_tb = tb_bands_on_kgrid(k_dimless, N, t, a, Hk_func)
        diff = E_tb[:, cidx] - E_pi_dft[:, cidx]
        return np.sqrt(np.mean(diff ** 2))

    res = minimize(loss, np.array([t0]), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4})
    t_fit = -abs(res.x[0])
    E_tb = tb_bands_on_kgrid(k_dimless, N, t_fit, a, Hk_func)
    rmse = loss([t_fit])
    return dict(t=t_fit, dE=0.0, rmse=rmse, E_tb=E_tb, central_indices=cidx,
                success=res.success, message=res.message)


def gaussian_dos(eigs: np.ndarray, E_grid: np.ndarray, eta: float):
    """
    eigs: (Nk, nb) eigenvalues
    returns DOS(E_grid)
    """
    flat = eigs.ravel()
    X = E_grid[:, None] - flat[None, :]
    G = np.exp(-0.5 * (X / eta) ** 2) / (np.sqrt(2 * np.pi) * eta)
    return G.sum(axis=1) / eigs.shape[0]


# ----------------------------------------
# -------------- Runner ------------------
# ----------------------------------------
def run_one_width(N: int, sc: SweepConfig, progress_desc: str = ""):
    out_dir = os.path.join(sc.out_root, f"ZGNR-{N}")
    _ensure_dir(out_dir)

    manifest = {
        "width": N,
        "label_base": f"zgnr{N}_nonmag",
        "files": {},
        "fits": {}
    }

    # Load TB module once here
    tb_mod = cfg.get_tb_module()
    Hk_func = tb_mod.H_zgnr_k

    # 1) Relax
    gpw_file, label = relax_zgnr_to_gpw(N, out_dir, sc)
    manifest["files"]["gpw"] = gpw_file

    # 2) Bands & DFT DOS
    k_dimless, E_rel_all, E_grid, DOS_dft, a_dft = bands_and_dos_from_gpw(gpw_file, label, out_dir, sc)
    manifest["files"]["dft_bands_dos_npz"] = os.path.join(out_dir, f"{label}_bands_dos.npz")

    # 3) π selection
    E_pi_dft, band_indices = select_pi_bands(E_rel_all, N)
    np.savez(
        os.path.join(out_dir, f"{label}_pi_bands.npz"),
        k_dimless=k_dimless,
        E_pi_dft=E_pi_dft,
        band_indices=band_indices,
        a_dft=a_dft,
    )
    manifest["files"]["pi_npz"] = os.path.join(out_dir, f"{label}_pi_bands.npz")

    # 4) TB fit: full π
    fit_full = fit_tb_full_pi(k_dimless, E_pi_dft, N, a_dft, Hk_func,
                              x0=(-2.7, 0.0), maxiter=sc.nm_maxiter_full)
    manifest["fits"]["full_pi"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                   for k, v in fit_full.items() if k != "E_tb"}
    np.savez(
        os.path.join(out_dir, f"{label}_tb_fit_full_pi.npz"),
        t=fit_full["t"], dE=fit_full["dE"], rmse=fit_full["rmse"],
        k_dimless=k_dimless, E_tb=fit_full["E_tb"], E_pi_dft=E_pi_dft, a_dft=a_dft
    )
    # plots: bands overlay
    plot_bands_png(
        k_dimless,
        bands_list=[
            (E_pi_dft, "-", 1.1, "C0", "DFT π"),
            (fit_full["E_tb"], "--", 1.1, "C1", "TB (full π fit)"),
        ],
        labels=None,
        title=f"{label}: DFT π vs TB (full π fit)",
        out_png=os.path.join(out_dir, f"{label}_dft_vs_tb_full_pi.png"),
    )
    if _HAS_PLOTLY:
        plot_bands_html(
            k_dimless,
            datasets=[("DFT π", E_pi_dft, "solid"), ("TB (full π fit)", fit_full["E_tb"], "dash")],
            title=f"{label}: DFT π vs TB (full π fit)",
            out_html=os.path.join(out_dir, f"{label}_dft_vs_tb_full_pi.html"),
        )

    # 5) TB fit: central-only
    fit_cen = fit_tb_central_only(k_dimless, E_pi_dft, N, a_dft, Hk_func,
                                  maxiter=sc.nm_maxiter_central, t0=-2.7)
    manifest["fits"]["central_only"] = {
        "t": float(fit_cen["t"]),
        "dE": float(fit_cen["dE"]),
        "rmse": float(fit_cen["rmse"]),
        "central_indices": [int(i) for i in fit_cen["central_indices"]],
        "success": bool(fit_cen["success"]),
        "message": fit_cen["message"],
    }
    np.savez(
        os.path.join(out_dir, f"{label}_tb_fit_central_only.npz"),
        t=fit_cen["t"], rmse=fit_cen["rmse"], central_indices=np.array(fit_cen["central_indices"]),
        k_dimless=k_dimless, E_tb=fit_cen["E_tb"], E_pi_dft=E_pi_dft, a_dft=a_dft
    )
    # plots: just central bands
    cidx = fit_cen["central_indices"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = k_dimless / np.pi
    ax.plot(x, E_pi_dft[:, cidx[0]], color="C0", lw=2.2, label="DFT π (valence)")
    ax.plot(x, E_pi_dft[:, cidx[1]], color="C1", lw=2.2, label="DFT π (conduction)")
    ax.plot(x, fit_cen["E_tb"][:, cidx[0]], "--", color="C0", lw=2.2, label="TB fit (valence)")
    ax.plot(x, fit_cen["E_tb"][:, cidx[1]], "--", color="C1", lw=2.2, label="TB fit (conduction)")
    ax.axhline(0.0, ls="--", lw=0.7, color="k", alpha=0.6)
    ax.set_xlabel(r"$k a / \pi$")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"{label}: Central π – DFT vs TB (central-only fit)")
    ax.legend(loc="best", fontsize=10)
    fig_save_png(fig, os.path.join(out_dir, f"{label}_pi_fit_central_ONLY.png"))
    if _HAS_PLOTLY:
        plot_bands_html(
            k_dimless,
            datasets=[
                ("DFT π central", E_pi_dft[:, cidx[0]:cidx[1]+1], "solid"),
                ("TB central", fit_cen["E_tb"][:, cidx[0]:cidx[1]+1], "dash"),
            ],
            title=f"{label}: Central π – DFT vs TB (central-only fit)",
            out_html=os.path.join(out_dir, f"{label}_pi_fit_central_ONLY.html"),
        )

    # 6) TB DOS for both fits (compare with DFT DOS)
    # full-π fit (use t and dE)
    ks_tb = np.linspace(0.0, np.pi / a_dft, sc.nk_tb_dos)
    E_tb_full = np.zeros((sc.nk_tb_dos, 2 * N))
    for i, k in enumerate(ks_tb):
        w, _ = np.linalg.eigh(Hk_func(k, N, t=fit_full["t"], a=a_dft))
        E_tb_full[i, :] = np.sort(w.real) + fit_full["dE"]
    DOS_tb_full = gaussian_dos(E_tb_full, E_grid, sc.eta_dos)

    # central-only fit DOS (use t only; no shift)
    E_tb_cen = np.zeros((sc.nk_tb_dos, 2 * N))
    for i, k in enumerate(ks_tb):
        w, _ = np.linalg.eigh(Hk_func(k, N, t=fit_cen["t"], a=a_dft))
        E_tb_cen[i, :] = np.sort(w.real)
    DOS_tb_cen = gaussian_dos(E_tb_cen, E_grid, sc.eta_dos)

    np.savez(
        os.path.join(out_dir, f"{label}_tb_dos.npz"),
        E_grid=E_grid,
        DOS_dft=DOS_dft,
        DOS_tb_full=DOS_tb_full,
        DOS_tb_central=DOS_tb_cen,
        t_full=fit_full["t"], dE_full=fit_full["dE"],
        t_central=fit_cen["t"],
    )

    plot_dos_png(
        E_grid,
        curves=[
            (DOS_dft, "DFT DOS", "-", 1.4),
            (DOS_tb_full, "TB DOS (full π fit)", "--", 1.3),
            (DOS_tb_cen, "TB DOS (central-only fit)", "-.", 1.1),
        ],
        title=f"DOS comparison – {label}",
        out_png=os.path.join(out_dir, f"{label}_dos_compare.png"),
    )
    if _HAS_PLOTLY:
        plot_dos_html(
            E_grid,
            curves=[
                (DOS_dft, "DFT DOS"),
                (DOS_tb_full, "TB DOS (full π)"),
                (DOS_tb_cen, "TB DOS (central)"),
            ],
            title=f"DOS comparison – {label}",
            out_html=os.path.join(out_dir, f"{label}_dos_compare.html"),
        )

    # Save a small MANIFEST per width
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    sc = SweepConfig()
    _ensure_dir(sc.out_root)

    all_manifests = []
    t0 = time.time()

    pbar = tqdm(sc.widths, desc="ZGNR sweep", ncols=90)
    for N in pbar:
        pbar.set_postfix_str(f"N={N}")
        manifest = run_one_width(N, sc)
        all_manifests.append(manifest)

    summary = {
        "config": asdict(sc),
        "manifests": all_manifests,
        "elapsed_s": round(time.time() - t0, 2),
        "note": "Interactive HTML plots are present only if plotly is installed."
    }
    with open(os.path.join(sc.out_root, "SWEEP_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Sweep done ===")
    for m in all_manifests:
        w = m["width"]
        fits = m["fits"]
        print(f" N={w:>2} | fullπ: t={fits['full_pi']['t']:.4f} eV, dE={fits['full_pi']['dE']:.4f}, "
              f"rmse={fits['full_pi']['rmse']:.4f} | central: t={fits['central_only']['t']:.4f}, "
              f"rmse={fits['central_only']['rmse']:.4f}")

if __name__ == "__main__":
    main()
