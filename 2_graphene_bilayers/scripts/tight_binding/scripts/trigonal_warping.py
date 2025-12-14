# trigonal_warping.py
import gpaw.new.ase_interface
import numpy as np
import matplotlib.pyplot as plt
from ase.dft.kpoints import BandPath, get_bandpath


def make_k_patch_around_K(atoms, radius=0.03, Nq=41, path=['M', 'K', 'G']):
    """
    Build a small 2D k-grid around the K-point in *scaled* coordinates.

    Parameters
    ----------
    atoms : ASE Atoms
        System used in the DFT calculation (bilayer AB graphene).
    radius : float
        Half-size of the k-patch in scaled coordinates.
        Typical interesting values: 0.01 – 0.05.
    Nq : int
        Number of points along each direction in the patch.
    path : list of str
        High-symmetry path including 'K', used only to locate K.

    Returns
    -------
    kpts_patch : (Nq*Nq, 3) array
        Scaled k-points around K.
    qx, qy : (Nq, Nq) arrays
        Displacement from K in scaled coordinates (for plotting).
    """

    # Use ASE to get the K point in scaled coords
    #bp = BandPath(path=path, cell=atoms.cell)
    #print(bp.special_points)
    print('going to generate bandpath')
    bp: BandPath = get_bandpath(path, atoms.cell)

    print('generated bandpath')
    K = bp.special_points['K']   # something like [1/3, 1/3, 0]

    # Build small square grid around K in the (kx, ky) plane
    q = np.linspace(-radius, radius, Nq)
    qx, qy = np.meshgrid(q, q, indexing='ij')

    kpts_patch = np.zeros((Nq * Nq, 3))
    kpts_patch[:, 0] = (K[0] + qx.ravel()) % 1.0
    kpts_patch[:, 1] = (K[1] + qy.ravel()) % 1.0
    kpts_patch[:, 2] = K[2]          # usually 0 for graphene

    return kpts_patch, qx, qy


def compute_patch_bands(calc: gpaw.new.ase_interface.ASECalculator, atoms, radius=0.03, Nq=41, nbands=None,
                        txt='tw_patch.txt'):
    """
    Fixed-density band calculation on a small k-patch around K.

    Returns energies shifted by EF.
    """

    from gpaw import GPAW  # imported here to avoid hard dependency at top

    kpts_patch, qx, qy = make_k_patch_around_K(atoms, radius=radius, Nq=Nq)

    # Fixed-density bandstructure on this patch
    # (re-uses SCF density from `calc`)
    params = dict(
        kpts=kpts_patch,
        symmetry='off',
        txt=txt
    )
    if nbands is not None:
        params['nbands'] = nbands

    print('Starting to compute the energies...')
    calc_patch = calc.fixed_density(**params)
    bs = calc_patch.band_structure()

    E = bs.energies[0]             # shape (Nk, Nbands)
    EF = calc_patch.get_fermi_level()

    E_shifted = E - EF
    Nbands = E.shape[1]

    # reshape to (Nq, Nq, Nbands) for easier plotting
    E_shifted = E_shifted.reshape(Nq, Nq, Nbands)

    return qx, qy, E_shifted, EF


def find_pi_bands(E_shifted, n=4):
    """
    Select n π-like bands closest to E = 0.

    Parameters
    ----------
    E_shifted : array (Nq, Nq, Nbands)
        Energies shifted so that EF = 0
    n : int
        Number of π bands to select (4 for bilayer graphene)

    Returns
    -------
    idx_pi : list of band indices
    E_pi : array (Nq, Nq, n)
        The n π bands across the full k-patch
    """
    # Average each band’s energy over the k-patch
    avg_E = E_shifted.mean(axis=(0, 1))  # shape (Nbands,)

    # Find the n bands closest to zero energy
    dist = np.abs(avg_E)
    idx_pi = np.argsort(dist)[:n]

    # Extract those bands
    E_pi = E_shifted[..., idx_pi]

    print(f'the closest energies to 0 are {idx_pi}, {E_pi}')

    return idx_pi, E_pi


def plot_trigonal_warping(qx, qy, E_band, E_iso=0.01,
                          title="Trigonal warping near K"):
    """
    Plot constant-energy contour of a low-energy band near K.

    Parameters
    ----------
    qx, qy : (Nq, Nq)
        Displacements from K in scaled coordinates.
    E_band : (Nq, Nq)
        Single band energies (shifted by EF).
    E_iso : float
        Target energy (in eV) for iso-contour (approx).
    """

    plt.figure(figsize=(6, 5))
    levels = [-E_iso, E_iso]  # small positive/negative energies
    cs = plt.contour(qx, qy, E_band, levels=levels, colors=['b', 'r'])
    plt.clabel(cs, inline=True, fontsize=8)

    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.gca().set_aspect('equal', 'box')

    plt.xlabel(r'$q_x$ (scaled around K)')
    plt.ylabel(r'$q_y$ (scaled around K)')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


import os
import plotly.graph_objects as go


def plot_trigonal_warping_3d(qx,
                             qy,
                             E_band,
                             title="Trigonal warping near K (3D)",
                             zlim=None,
                             save_html=True,
                             html_path="../data/trigonal_warping_3d.html"):
    """
    Interactive 3D surface plot of a single low-energy band near K.

    Parameters
    ----------
    qx, qy : (Nq, Nq) arrays
        Displacements from K in scaled coordinates (meshgrid).
    E_band : (Nq, Nq) array
        Energies of one band, already shifted so EF = 0.
    title : str
        Plot title.
    zlim : tuple or None
        (zmin, zmax) to clip energy range for visualization, e.g. (-0.05, 0.05).
    save_html : bool
        If True, save interactive HTML plot.
    html_path : str
        Path where the HTML file will be saved.
    """

    Z = E_band.copy()
    if zlim is not None:
        zmin, zmax = zlim
        Z = Z.clip(zmin, zmax)

    fig = go.Figure(data=[
        go.Surface(
            x=qx,
            y=qy,
            z=Z,
            showscale=True,
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="kx",
            yaxis_title="ky",
            zaxis_title="E - E_F (eV)",
            aspectmode="cube",
        ),
        template="plotly_white",
    )

    if save_html:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"Saved interactive 3D trigonal-warping plot to: {html_path}")

    fig.show()
    return fig