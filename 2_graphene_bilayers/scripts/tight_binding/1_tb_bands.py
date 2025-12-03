import numpy as np
import matplotlib.pyplot as plt

# --- Tight-binding parameters (eV) ---
gamma0 = 3.1      # in-plane NN
gamma1 = 0.39     # vertical (dimer) hopping
gamma3 = 0.315    # skew interlayer A1 <-> B2
gamma4 = 0.044    # skew same-sublattice A1 <-> A2 , B1 <-> B2
# On-site energies
epsA1 = 0.0
epsB1 = 0.0
epsA2 = 0.0
epsB2 = 0.0

# --- Graphene lattice constants ---
a = 1.42  # C-C distance (Å)
a0 = np.sqrt(3) * a  # lattice constant ~ 2.46 Å

# nearest neighbour vectors in real space
delta = np.array([
    [0,  a/np.sqrt(3)],
    [ a/2, -a/(2*np.sqrt(3))],
    [-a/2, -a/(2*np.sqrt(3))]
])


def f_k(kx, ky):
    """Monolayer structure factor f(k)."""
    return np.sum(np.exp(1j * (delta[:,0]*kx + delta[:,1]*ky)))


def make_H(kx, ky):
    """Return 4x4 bilayer graphene Hamiltonian H(k)."""

    f = f_k(kx, ky)
    f_conj = np.conj(f)

    # Build the Hamiltonian
    H = np.zeros((4,4), dtype=complex)

    # Diagonals
    H[0,0] = epsA1
    H[1,1] = epsB1
    H[2,2] = epsA2
    H[3,3] = epsB2

    # Intralayer gamma0
    H[0,1] = -gamma0 * f
    H[1,0] = -gamma0 * f_conj
    H[2,3] = -gamma0 * f
    H[3,2] = -gamma0 * f_conj

    # Interlayer gamma1 (vertical)
    H[1,2] = gamma1
    H[2,1] = gamma1

    # Skew interlayer gamma3
    H[0,3] = -gamma3 * f_conj
    H[3,0] = -gamma3 * f

    # Skew interlayer gamma4 (same sublattice)
    H[0,2] = gamma4 * f
    H[2,0] = gamma4 * f_conj
    H[1,3] = gamma4 * f
    H[3,1] = gamma4 * f_conj

    return H


# --- High-symmetry path Γ → K → M → Γ ---
# Reciprocal lattice vectors
b1 = (2*np.pi/a0) * np.array([1, 1/np.sqrt(3)])
b2 = (2*np.pi/a0) * np.array([1,-1/np.sqrt(3)])

# High symmetry points
Gamma = np.array([0,0])
K     = (b1 + 2*b2) / 3
M     =  b1 / 2

# Build the k-path
N = 200
k_path = []
label_positions = []
labels = [r"$\Gamma$", "K", "M", r"$\Gamma$"]

# Γ → K
for t in np.linspace(0,1,N):
    k_path.append((1-t)*Gamma + t*K)
label_positions.append(0)

# K → M
offset = len(k_path)
for t in np.linspace(0,1,N):
    k_path.append((1-t)*K + t*M)
label_positions.append(offset)

# M → Γ
offset2 = len(k_path)
for t in np.linspace(0,1,N):
    k_path.append((1-t)*M + t*Gamma)
label_positions.append(offset2)
label_positions.append(len(k_path)-1)

k_path = np.array(k_path)

# --- Compute eigenvalues ---
bands = []
for kx, ky in k_path:
    H = make_H(kx, ky)
    eigs = np.linalg.eigvalsh(H)
    bands.append(eigs)

bands = np.array(bands)


# --- Plot ---
plt.figure(figsize=(6,5))
for i in range(4):
    plt.plot(bands[:,i], lw=2)

# vertical lines & labels
for p in label_positions:
    plt.axvline(p, color='gray', linewidth=0.5)

plt.xticks(label_positions, labels)
plt.ylabel("Energy (eV)")
plt.title("Bilayer Graphene Band Structure (Tight Binding)")
plt.tight_layout()
plt.show()

print(k_path)