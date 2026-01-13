from __future__ import annotations

# Note: No fixed_density or special Poisson solver is used at this stage.


def apply_external(calc0, atoms, ext, txt: str = "gate.txt"):
    """Apply external potential and run a regular SCF on the original calculator.

    This mirrors your external_field.ipynb workflow:
    - Set the external potential on the existing calculator (no fixed_density here)
    - Attach to atoms; caller triggers SCF with atoms.get_potential_energy()

    Parameters
    ----------
    calc0 : gpaw.GPAW
        A converged GPAW calculator (e.g., from restart).
    atoms : ase.Atoms
        The system to which the calculator will be re-attached.
    ext : gpaw.external.ExternalPotential
        External potential object with ``calculate_potential`` method.
    txt : str
        GPAW txt log file name for the SCF with the external potential.

    Returns
    -------
    gpaw.GPAW
        The same calculator with the external potential applied.
    """

    calc = calc0
    # Set external potential and output txt; do not use fixed_density here
    calc.set(external=ext, txt=txt)

    atoms.calc = calc
    return calc
