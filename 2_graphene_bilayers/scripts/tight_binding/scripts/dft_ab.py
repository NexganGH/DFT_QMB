from ase import Atoms
import numpy as np
from gpaw import GPAW, PW
import os.path

def run_dft_ab(filename='DFT.gpw') -> None:
    '''
    Runs a DFT calculation for graphene AB bilayer and saves it in ../gpw/`filename`.gpw
    :param filename: GPW Filename
    :return: None
    '''
    path = '../' + filename
    if os.path.isfile(path):
        raise 'Specified already exists! Delete it or change filename'

    a0 = 2.46
    b  = a0 / np.sqrt(3)
    d  = 3.35

    # Coordinates for BA stacking
    positions = [
        (0.0,      0.0,   0.0),   # A1
        (a0/2,     b/2,   0.0),   # B1
        (-a0/2,   -b/2,   d),     # A2
        (0.0,      0.0,   d),     # B2 (directly above A1)
    ]

    atoms = Atoms(
        "C4",
        positions=positions,
        cell=[
            [a0,     0.0,    0.0],
            [a0/2,  1.5*b,   0.0],
            [0.0,    0.0,  10*d],
        ],
        pbc=True,
    )

    # --- COLORING ---
    # lower layer: red
    # upper layer: blue
    colors = [
        (1.0, 0.0, 0.0),  # A1 red
        (1.0, 0.0, 0.0),  # B1 red
        (0.0, 0.0, 1.0),  # A2 blue
        (0.0, 0.0, 1.0),  # B2 blue
    ]

    atoms.set_array('colors', np.array(colors))

    calc = GPAW(
        mode=PW(600),  # 600 eV cutoff (you can reduce to 400 if needed)
        xc='PBE',
        kpts=(18, 18, 1),
        occupations={'name': 'fermi-dirac', 'width': 0.01},
        txt='DFT.txt',
    )

    atoms.calc = calc
    E = atoms.get_potential_energy()
    print(f'Total energy: {E:.6f} eV')

    # Save full ground state for band-structure run
    calc.write(path, mode='all')
    print(f'File saved to {path}')
