import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load NPZ file with extracted π bands from GPAW
# ---------------------------------------------------------
data = np.load("pi_bands_from_gpaw.npz")

kpts = data["kpts"]        # shape (Nk, 2)
E_pi = data["energies"]    # shape (Nk, 4)
idx_pi = data["band_indices"]    # π band indices in DFT
EF = data["EF"]

Nk = kpts.shape[0]
npi = E_pi.shape[1]

print("Loaded π bands:", idx_pi)
print("Number of k-points:", Nk)

# ---------------------------------------------------------
# Build x-axis from k-point path (cumulative distance)
# ---------------------------------------------------------
# k distance for plotting
dk = np.sqrt(np.sum((np.diff(kpts, axis=0) ** 2), axis=1))
x = np.concatenate([[0], np.cumsum(dk)])

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

for i in range(npi):
    plt.plot(x, E_pi[:, i], label=f"π band {idx_pi[i]}")

plt.axhline(0.0, color="k", lw=0.5)  # EF

# ---------------------------------------------------------
# High-symmetry labels (depends on how you built path)
# Example: M – K – Γ with 200 points total
# ---------------------------------------------------------
# You can manually mark the boundaries:
# Example (if kpts were generated with 200 points from M→K→Γ):
#  M (x=0), K (x=100), G (x=200)
# Adjust if you used a different number.
plt.xticks([x[0], x[Nk//2], x[-1]], ["M", "K", "Γ"])

plt.xlabel("k-path")
plt.ylabel("Energy (eV, shifted to EF=0)")
plt.title("Extracted DFT π Bands")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
