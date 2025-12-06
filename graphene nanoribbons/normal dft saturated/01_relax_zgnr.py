# 01_relax_zgnr.py
### ATTENTION: WE HAVE NON-SATURATED EDGES HERE.
### IN THE SPIN-POLARISED CALCULATION WE USE SATURATED EDGES.

from ase.build import graphene_nanoribbon
from ase.optimize import BFGS
from ase.io import write
from gpaw import GPAW, PW
import numpy as np

from config_zgnr import (
    N_ZIGZAG,
    LENGTH_REPEATS,
    VACUUM,
    LABEL_BASE,
    KPTS_1D_RELAX,
    ECUT,
    FMAX,
)


def make_zgnr(N_zigzag, length_repeats, vacuum):
    """
    Create a non-magnetic zigzag graphene nanoribbon
    with NON-saturated edges (no hydrogen).
    """
    atoms = graphene_nanoribbon(
        N_zigzag,
        length_repeats,
        type="zigzag",
        saturated=True,   # <- NON-saturated edges as desired
        vacuum=vacuum,
        magnetic=False,
    )

    # IMPORTANT: use full PBC for GPAW when using k-points
    atoms.pbc = (True, True, True)

    # 1) Put the ribbon roughly in the middle of the cell
    atoms.center()  # keeps the cell, just translates positions

    # 2) In scaled coords, push every atom away from all cell boundaries
    #    This avoids "Some atom is too close to the zero-boundary!" in GPAW.
    spos = atoms.get_scaled_positions()  # shape (N_atoms, 3), in [0, 1)
    eps = 0.05  # 5% margin from each boundary

    # Clip all three components to [eps, 1-eps]
    spos = np.clip(spos, eps, 1.0 - eps)
    atoms.set_scaled_positions(spos)

    # (Optional sanity check)
    # print("Scaled mins:", spos.min(axis=0))
    # print("Scaled maxs:", spos.max(axis=0))

    return atoms


def relax_zgnr():
    """
    Relax the ZGNR structure and save the relaxed state to a gpw file.
    """
    atoms = make_zgnr(N_ZIGZAG, LENGTH_REPEATS, VACUUM)

    calc = GPAW(
        mode=PW(ECUT),
        xc="PBE",
        kpts=(1, 1, KPTS_1D_RELAX),
        spinpol=False,
        txt=f"{LABEL_BASE}_relax.txt",
    )
    atoms.calc = calc

    dyn = BFGS(
        atoms,
        trajectory=f"{LABEL_BASE}_relax.traj",
        logfile=f"{LABEL_BASE}_relax.log",
    )
    dyn.run(fmax=FMAX)

    # Save relaxed geometry and gpw
    write(f"{LABEL_BASE}_relaxed.xyz", atoms)
    gpw_file = f"{LABEL_BASE}_relaxed.gpw"
    calc.write(gpw_file, mode="all")

    print(f"Relaxation done. Saved {gpw_file}")
    return gpw_file


if __name__ == "__main__":
    relax_zgnr()
