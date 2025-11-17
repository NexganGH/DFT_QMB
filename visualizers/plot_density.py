from ase.io import read
import numpy as np

atoms = read("graphene_density.cube")

# volumetric grid (rho[x,y,z])
rho = atoms.calc.data['data']     # ndarray

# origin of grid
origin = atoms.calc.data['origin']

# number of grid points (Nx, Ny, Nz)
N = atoms.calc.data['N']

# grid vectors
xvec = atoms.calc.data['xvec']
yvec = atoms.calc.data['yvec']
zvec = atoms.calc.data['zvec']
