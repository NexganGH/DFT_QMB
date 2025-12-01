from gpaw.mpi import world
from gpaw import GPAW, FermiDirac
from ase.visualize import view
from ase.io import Trajectory

if __name__ == '__main__':
    from ase.build import graphene_nanoribbon
    atoms = graphene_nanoribbon(
        n=8,
        m=2,
        type='zigzag',
        saturated=True,
        C_C=1.42,
        C_H=1.09,
        vacuum=10.0,
        sheet=False
    )
    calc = GPAW(
        mode='lcao',  # Fast for relaxation; switch to 'pw' for final energies
        basis='dzp',  # Good compromise for C systems
        xc='PBE',  # Standard GGA
        spinpol=True,  # ❤️ Enable spin-polarisation
        occupations=FermiDirac(0.10),  # Smearing (small)
        txt='gnr_relax.txt',
        kpts=(1, 1, 3),  # period along z; adjust depending on orientation
    )
    print('Starting DFT calculation...')
    atoms.calc = calc

    from ase.optimize import BFGS

    dyn = BFGS(atoms, trajectory='gnr_relax.traj', )
    dyn.run(fmax=0.02)

    view(Trajectory('gnr_relax.traj'))

    calc.write('gnr_relaxed.gpw', mode='all')
