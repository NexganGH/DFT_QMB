from scipy.optimize import least_squares
import numpy as np
from tb_bands import tb_bands
import  matplotlib.pyplot as plt


# kpts: (Nk, 2)
# E_dft_pi: (Nk, 4)  # the four π-like bands
# maybe shift so that Fermi level is at 0
#E_dft_shifted = E_dft_pi - E_dft_pi.mean(axis=(0,1))  # or subtract EF

data = np.load("pi_bands_from_gpaw.npz")

kpts = data["kpts"]        # shape (Nk, 2)
E_pi = data["energies"]    # shape (Nk, 4)
idx_pi = data["band_indices"]    # π band indices in DFT
EF = data["EF"]

E_dft_shifted = E_pi
cost_history = []
def cost_function(params, kpts, E_dft):
    """Return flatten residual vector (least_squares format)."""
    global cost_history

    E_tb = tb_bands(kpts, params)        # (Nk, 4)
    residual = (E_tb - E_dft).ravel()

    # Record cost = ||residual||^2
    cost = np.sum(residual**2)
    cost_history.append(cost)

    return residual


# initial guess from literature
theta0 = np.array([
    2.7,   # gamma0
    0.39,  # gamma1
    0.3,   # gamma3
    0.04,  # gamma4
    0.0, 0.0, 0.0, 0.0  # epsA1, epsB1, epsA2, epsB2
])

res = least_squares(cost_function,
                    theta0,
                    args=(kpts, E_pi),
                    max_nfev=200,
                    verbose=2)

theta_fit = res.x
print("Fitted parameters:", theta_fit)

plt.figure(figsize=(7,4))
plt.plot(cost_history, marker="o")
#plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("TB Fit Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

E_tb = tb_bands(kpts, theta_fit)

# =====================================================
# 5. BUILD X-AXIS (distance along path)
# =====================================================

dk = np.sqrt(np.sum(np.diff(kpts, axis=0)**2, axis=1))
x = np.concatenate([[0], np.cumsum(dk)])

#
# 6. PLOT BEGINNING
#
E_tb_0 = tb_bands(kpts, theta0)


plt.figure(figsize=(8,5))

nb = E_pi.shape[1]

# DFT π bands
for i in range(nb):
    plt.plot(x, E_pi[:, i], 'k-', lw=2, label="DFT π bands" if i==0 else "")

# TB π bands
for i in range(4):
    plt.plot(x, E_tb_0[:, i], 'r--', lw=1.5, label="TB fit" if i==0 else "")

plt.axhline(0, color='gray', lw=0.5)
plt.xlabel("k-path")
plt.ylabel("Energy (eV, EF=0)")
plt.title("DFT π Bands vs Tight-Binding Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# 6. PLOT
# =====================================================
plt.figure(figsize=(8,5))

nb = E_pi.shape[1]

# DFT π bands
for i in range(nb):
    plt.plot(x, E_pi[:, i], 'k-', lw=2, label="DFT π bands" if i==0 else "")

# TB π bands
for i in range(4):
    plt.plot(x, E_tb[:, i], 'r--', lw=1.5, label="TB fit" if i==0 else "")

plt.axhline(0, color='gray', lw=0.5)
plt.xlabel("k-path")
plt.ylabel("Energy (eV, EF=0)")
plt.title("DFT π Bands vs Tight-Binding Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()