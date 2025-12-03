import numpy as np
import matplotlib.pyplot as plt

# Tight-binding parameters (eV)
gamma0 = 3.1
gamma1 = 0.39
gamma3 = 0.315
gamma4 = 0.044

# Lattice constants
a = 1.42
a0 = np.sqrt(3) * a

# Nearest-neighbour vectors
delta = np.array([
    [0,  a/np.sqrt(3)],
    [ a/2, -a/(2*np.sqrt(3))],
    [-a/2, -a/(2*np.sqrt(3))]
])

def f_k(kx, ky):
    return np.sum(np.exp(1j*(delta[:,0]*kx + delta[:,1]*ky)))

def make_H(kx, ky):
    f = f_k(kx, ky)
    H = np.zeros((4,4), dtype=complex)

    H[0,1] = -gamma0 * f
    H[1,0] = -gamma0 * np.conj(f)
    H[2,3] = -gamma0 * f
    H[3,2] = -gamma0 * np.conj(f)

    H[1,2] = gamma1
    H[2,1] = gamma1

    H[0,3] = -gamma3 * np.conj(f)
    H[3,0] = -gamma3 * f

    H[0,2] = gamma4 * f
    H[2,0] = gamma4 * np.conj(f)
    H[1,3] = gamma4 * f
    H[3,1] = gamma4 * np.conj(f)

    return H

# Sample dense k-grid for DOS
Nk = 150
kx_vals = np.linspace(-2*np.pi/a0, 2*np.pi/a0, Nk)
ky_vals = np.linspace(-2*np.pi/a0, 2*np.pi/a0, Nk)

energies = []

for kx in kx_vals:
    for ky in ky_vals:
        H = make_H(kx, ky)
        eigs = np.linalg.eigvalsh(H)
        energies.extend(eigs)

energies = np.array(energies)

# Compute DOS via histogram
dos_bins = 400
dos, E_bins = np.histogram(energies, bins=dos_bins, density=True)
E_centers = 0.5 * (E_bins[:-1] + E_bins[1:])

# Plot DOS
plt.figure(figsize=(7,5))
plt.plot(E_centers, dos)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS (arb. units)")
plt.title("Density of States of Bilayer Graphene (TB Model)")
plt.grid(True)
plt.show()
