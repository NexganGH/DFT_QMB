import numpy as np
from gpaw.external import ExternalPotential


class ChargedPlanePotential(ExternalPotential):
    """Linear potential produced by a uniformly charged plane at z=z_plane.

    V(z) = -A * |z - z_plane|, with mean removed to avoid constant offset.

    Parameters
    ----------
    A : float
        Field strength in eV/Å.
    z_plane : float
        Plane position along z (Å).
    """

    def __init__(self, A: float, z_plane: float):
        self.A = float(A)
        self.z_plane = float(z_plane)

    def calculate_potential(self, gd):
        r_vg = gd.get_grid_point_coordinates()
        z = r_vg[2]
        V = -self.A * np.abs(z - self.z_plane)
        # remove constant offset as recommended
        self.vext_g = V - V.mean()

    def todict(self):
        return {
            'name': 'ChargedPlanePotential',
            'A': self.A,
            'z_plane': self.z_plane,
        }


def register_with_gpaw():
    """Register this custom potential class with GPAW's known potentials.

    GPAW writes ``vext`` to the GPW file as a small dict. On restart, it
    attempts to reconstruct the object via ``gpaw.external.create_external_potential``
    which looks up the class in ``known_potentials`` by name. If the class
    isn't registered, a KeyError occurs. Call this once per Python process
    before using ``restart(...)`` to read GPWs that include this potential.
    """
    try:
        from gpaw.external import known_potentials
        known_potentials['ChargedPlanePotential'] = ChargedPlanePotential
    except Exception:
        # Be silent if GPAW API changes; the notebook can still set external explicitly
        pass
