from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

import numpy as np
from gpaw import GPAW

from external_potentials import ChargedPlanePotential
from gpaw_hooks import apply_external


def compute_gap(E_shifted_kb: np.ndarray) -> float:
    """Compute band gap from energies shifted by Fermi level.

    Parameters
    ----------
    E_shifted_kb : (Nk, Nb) array
        Energies along a k-path with Fermi level subtracted.

    Returns
    -------
    float
        Gap in eV. If no positive/negative states found, returns np.nan.
    """
    if E_shifted_kb.size == 0:
        return float('nan')

    # Over k, get band extrema
    Emin_b = E_shifted_kb.min(axis=0)
    Emax_b = E_shifted_kb.max(axis=0)

    if not np.any(Emin_b > 0) or not np.any(Emax_b < 0):
        return float('nan')

    cbm = np.min(Emin_b[Emin_b > 0])
    vbm = np.max(Emax_b[Emax_b < 0])
    return float(cbm - vbm)


def band_structure_with_path(calc: GPAW,
                             path: str = 'GKMG',
                             npoints: int = 30,
                             nbands: Optional[int] = None,
                             convergence_bands: Optional[int] = None):
    """Convenience wrapper to get band structure along a k-path.

    Uses calculator.fixed_density to avoid re-doing SCF.
    """
    kwargs = dict(symmetry='off', kpts={'path': path, 'npoints': npoints})
    if nbands is not None:
        kwargs['nbands'] = nbands
    if convergence_bands is not None:
        kwargs['convergence'] = {'bands': convergence_bands}

    bs_calc = calc.fixed_density(**kwargs)
    return bs_calc.band_structure()


def sweep_field_gaps(atoms,
                     calc0: GPAW,
                     z_plane: float,
                     fields: Iterable[float],
                     path: str = 'GKMG',
                     npoints: int = 30,
                     nbands: Optional[int] = 16,
                     convergence_bands: Optional[int] = 8) -> List[Tuple[float, float]]:
    """Sweep external field strengths and compute the band gap for each.

    Returns list of (field, gap_eV).
    """
    results: List[Tuple[float, float]] = []
    for Ez in fields:
        ext = ChargedPlanePotential(A=Ez, z_plane=z_plane)
        calcE = apply_external(calc0, atoms, ext, txt=f'gate_A{Ez:.3f}.txt')

        # Run a regular SCF with the external potential applied
        _ = atoms.get_potential_energy()

        bs = band_structure_with_path(calcE, path=path, npoints=npoints,
                                       nbands=nbands, convergence_bands=convergence_bands)
        energies = bs.energies[0]  # (Nk, Nb)
        EF = calcE.get_fermi_level()
        E = energies - EF

        gap = compute_gap(E)
        results.append((Ez, float(gap)))

    return results
