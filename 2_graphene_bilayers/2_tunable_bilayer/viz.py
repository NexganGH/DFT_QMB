from __future__ import annotations

from typing import Tuple

import numpy as np


from typing import Tuple
import numpy as np
from common.mpl_style import set_mpl_style

def compute_external_grid(
    calc,
    ext,
    atoms,
    margin: float = 2.0,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute external potential on calculator's grid using ``ext``,
    restricted to a region around the atoms.

    Parameters
    ----------
    calc : GPAW calculator
    ext : external potential object
    atoms : ASE Atoms
    margin : float
        Extra margin (in Å) added around atomic bounding box.

    Returns
    -------
    Vext_cut : ndarray
        External potential on reduced grid.
    (x_cut, y_cut, z_cut) : tuple of ndarrays
        Coordinate arrays on reduced grid.
    """

    # --- full GPAW grid ---
    gd = calc.hamiltonian.xc_gd
    ext.calculate_potential(gd)
    Vext = ext.vext_g

    x, y, z = gd.get_grid_point_coordinates()

    # --- atomic bounding box ---
    pos = atoms.get_positions()
    xmin, ymin, zmin = pos.min(axis=0) - margin
    xmax, ymax, zmax = pos.max(axis=0) + margin

    # --- grid masks (use 1D representatives) ---
    x1d = x[:, 0, 0]
    y1d = y[0, :, 0]
    z1d = z[0, 0, :]

    ix = np.where((x1d >= xmin) & (x1d <= xmax))[0]
    iy = np.where((y1d >= ymin) & (y1d <= ymax))[0]
    iz = np.where((z1d >= zmin) & (z1d <= zmax))[0]

    # --- slice everything consistently ---
    Vext_cut = Vext[np.ix_(ix, iy, iz)]
    x_cut = x[np.ix_(ix, iy, iz)]
    y_cut = y[np.ix_(ix, iy, iz)]
    z_cut = z[np.ix_(ix, iy, iz)]

    return Vext_cut, (x_cut, y_cut, z_cut)



def plot_potential_slice(Vext: np.ndarray,
                         xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         atoms,
                         average_axis: int = 1,
                         cmap: str = 'coolwarm',
                         figsize=(7, 5),
                         title: str = 'External potential'):
    """Plot a 2D slice of the 3D external potential averaged over one axis.

    Parameters
    ----------
    Vext : array
        External potential on the grid (as produced by GPAW/ExternalPotential).
    xyz : tuple of arrays
        Grid coordinate arrays (x, y, z) as returned by ``compute_external_grid``.
    atoms : ase.Atoms
        Atoms to overlay their positions.
    average_axis : int
        Axis to average over: 0=x, 1=y, 2=z. Default averages over y.
    """
    import matplotlib.pyplot as plt
    set_mpl_style(fontsize=17)

    x, y, z = xyz

    # average over the chosen axis
    V_avg = Vext.mean(axis=average_axis)

    # choose axes for display; default to x–z plane
    x_vals = x[:, 0, 0]
    z_vals = z[0, 0, :]

    plt.figure(figsize=figsize)
    plt.pcolormesh(z_vals, x_vals, V_avg, shading='auto', cmap=cmap)
    plt.colorbar(label=r'$V_{ext}$ (eV)')
    plt.xlabel('z (Å)')
    plt.ylabel('x (Å)')
    plt.title(title)
    z_plane = 6.35  # Å (absolute Cartesian coordinate)

    # overlay atoms
    # --- shift atoms into grid coordinates ---
    x0 = x_vals.min()
    z0 = z_vals.min()

    pos = atoms.positions
    x_atoms = pos[:, 0] + 2
    z_atoms = pos[:, 2] - z0

    plt.scatter(
        z_atoms, x_atoms,
        color='white',
        s=200,
        edgecolors='black',
        linewidths=1.0,
        zorder=10,
        label='Atoms'
    )

    # --- draw charged plane position ---
    z_plane = 6.35  # Å (absolute Cartesian)
    z_plane_plot = z_plane - z0

    plt.axvline(
        z_plane_plot,
        color='black',
        linestyle='--',
        linewidth=3.5,  # thicker
        alpha=1.0,
        zorder=9,
        label=r'Charged plane'
    )

    plt.legend(loc='upper left')
    plt.tight_layout()
    return plt.gca()


def plot_band_structure(bs, title: str = 'Band structure (E - E_F)'):
    """Simple band structure plot subtracting Fermi level from energies.

    Parameters
    ----------
    bs : gpaw.response.band_structure.BandStructure
        BandStructure object from GPAW.
    """
    import matplotlib.pyplot as plt

    energies = bs.energies[0]
    # bs may not carry calculator; caller should shift if needed. Try best effort.
    EF = getattr(getattr(bs, 'calc', None), 'get_fermi_level', lambda: 0.0)()
    E_shift = energies - EF

    plt.figure(figsize=(6, 4))
    for n in range(E_shift.shape[1]):
        plt.plot(E_shift[:, n])

    plt.ylim(-1, 1)
    plt.xlabel('k-path index')
    plt.ylabel('Energy (eV)')
    plt.title(title)
    plt.tight_layout()
    return plt.gca()
