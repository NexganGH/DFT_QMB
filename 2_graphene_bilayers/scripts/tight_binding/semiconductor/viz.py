from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_external_grid(calc, ext) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute external potential on calculator's grid using ``ext``.

    Returns (Vext, (x, y, z)) arrays in GPAW grid ordering.
    """
    gd = calc.hamiltonian.xc_gd
    ext.calculate_potential(gd)
    Vext = ext.vext_g
    r = gd.get_grid_point_coordinates()
    x, y, z = r
    return Vext, (x, y, z)


def plot_potential_slice(Vext: np.ndarray,
                         xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         atoms,
                         average_axis: int = 1,
                         cmap: str = 'coolwarm',
                         figsize=(7, 5),
                         title: str = 'External potential + atom positions'):
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

    x, y, z = xyz

    # average over the chosen axis
    V_avg = Vext.mean(axis=average_axis)

    # choose axes for display; default to x–z plane
    x_vals = x[:, 0, 0]
    z_vals = z[0, 0, :]

    plt.figure(figsize=figsize)
    plt.pcolormesh(z_vals, x_vals, V_avg, shading='auto', cmap=cmap)
    plt.colorbar(label='V_ext (eV)')
    plt.xlabel('z (Å)')
    plt.ylabel('x (Å)')
    plt.title(title)

    # overlay atoms
    pos = atoms.positions
    x_atoms = pos[:, 0]
    z_atoms = pos[:, 2]
    plt.scatter(z_atoms, x_atoms, color='white', s=40, edgecolors='black', label='Atoms')

    plt.legend(loc='upper right')
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
