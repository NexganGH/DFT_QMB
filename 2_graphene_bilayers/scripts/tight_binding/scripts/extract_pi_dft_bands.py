from gpaw import restart
import numpy as np
from ase.dft.kpoints import get_bandpath
from ase.atoms import Atoms
from gpaw.new.ase_interface import ASECalculator
from ase.dft.kpoints import BandPath
from typing import Tuple
import os
from ase.dft.kpoints import kpoint_convert


def save_pi_dft_bands(atoms: Atoms, calc: ASECalculator, path=None,
                      npoints=50, zoom_label=None, zoom_distance=0.5,
                      filename="../data/gpaw_pi_bands.npz") -> None:
    """
    Save π-band energies derived from a DFT calculation and optionally prepare
    them for further Tight-Binding (TB) fitting. This function evaluates the
    energies of the four π bands closest to the Fermi energy from a band structure
    calculation. The results include k-point coordinates, π-band energies, and
    corresponding band indices, all shifted with respect to the Fermi level. The
    data is saved to a `.npz` file for further use.

    :param path:
    :param zoom_distance:
    :param zoom_label:
    :param atoms:
        An `Atoms` object containing the atomic structure and associated
        information such as the simulation cell.

    :param calc:
        A calculator object implementing the ASE calculator interface.
        It should be capable of performing DFT calculations and generating
        the band structure.

    :param kpts:
        Optional. Defines a set of k-points to be used; if not provided,
        this will automatically be set based on the bandpath of the system.

    :param npoints:
        Optional. Number of the k-points to be used.

    :param filename:
        Optional. Path to the output .npz file. Defaults to "../data/gpaw_pi_bands.npz"

    :return:
        None. Outputs from the band structure calculation are saved to the specified file.
    """
    if path is None:
        path = ['G', 'M', 'K', 'G']
    bandpath: BandPath = get_bandpath(path, atoms.cell, npoints=npoints)#kpts, x, X = get_bandpath(path, atoms.cell, npoints=200)
    kpts: np.ndarray = bandpath.kpts
    x, X, labels = bandpath.get_linear_kpoint_axis()
    kpts_cart = kpoint_convert(atoms.cell, skpts_kc=kpts)

    if zoom_label is not None:
        if zoom_label not in path: raise (f'The provided zoom_label zoom_label={zoom_label} is not in the '
                                          f'provided path {path}')
        special_point = kpoint_convert(atoms.cell, skpts_kc=bandpath.special_points[zoom_label])
        # To compute the distance, must use cartesian coordinates

        dist = np.linalg.norm(kpts_cart - special_point, axis=1)
        filter = dist < zoom_distance/2
        kpts = kpts[filter]
        kpts_cart = kpts_cart[filter]
        print(f'Going to get distance {zoom_distance} around {zoom_label}')
        print(f'Current kpath {kpts}')

    #atoms, calc = restart('DFT.gpw')

    calc_bs = calc.fixed_density(
        kpts=kpts,
        symmetry='off',
        txt='blg_bands_fixed_density.txt',
    )

    bs = calc_bs.band_structure()
    #kpts_cart = kpoint_convert(atoms.cell, skpts_kc=kpts)#bs.get_kpoints(cartesian=True)
    print(kpts_cart)
    # Energies and Fermi level
    E = bs.energies[0]        # shape: (Nk, Nbands)
    EF = calc_bs.get_fermi_level()

    # --- find π bands: 4 bands closest to EF ---
    # (2 π bands per layer → 4 for bilayer)
    dist = np.abs(E.mean(axis=0) - EF)   # distance of each band from EF
    idx_pi = np.argsort(dist)[:4]        # indices of π bands

    print("π-band indices:", idx_pi)

    # Extract π-band energies (shifted so EF = 0)
    E_pi = E[:, idx_pi] - EF             # shape: (Nk, 4)

    # Optionally save for TB fitting
    np.savez(filename,
             kpts=kpts_cart[:, :2],  # kx, ky
             energies=E_pi,
             band_indices=idx_pi,
             EF=EF)


def load_pi_dft_bands() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Loads the density functional theory (DFT) pi bands data from a specified file.

    This function retrieves the k-points, energy values, band indices, and Fermi 
    energy from a pre-saved file containing DFT pi band calculations. The data 
    is loaded from a NumPy `.npz` file located at a fixed path.

    :return: A tuple containing:
        - kpts (numpy.ndarray): An array of k-points for the calculation.
        - energies (numpy.ndarray): An array of energy values corresponding to
          the k-points.
        - band_indices (numpy.ndarray): An array containing indices of the bands.
        - EF (float): The Fermi energy value.
    """
    file_path = "../data/gpaw_pi_bands.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    data = np.load(file_path)
    return data["kpts"], data["energies"], data["band_indices"], data["EF"]