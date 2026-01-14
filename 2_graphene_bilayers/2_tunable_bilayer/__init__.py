"""Utilities for semiconductor external field studies with GPAW.

This package contains small, focused modules that can be imported from
notebooks while keeping heavy logic out of the notebook cells.
"""

from .external_potentials import ChargedPlanePotential  # re-export for convenience
from .gpaw_hooks import apply_external
from .analysis import compute_gap, sweep_field_gaps
from .viz import plot_potential_slice, plot_band_structure

__all__ = [
    "ChargedPlanePotential",
    "apply_external",
    "compute_gap",
    "sweep_field_gaps",
    "plot_potential_slice",
    "plot_band_structure",
]
