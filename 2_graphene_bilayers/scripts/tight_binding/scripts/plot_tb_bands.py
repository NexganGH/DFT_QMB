from tb_bands import tb_bands
import matplotlib.pyplot as plt
import numpy as np

theta0 = np.array([
    3.1,   # gamma0
    0.39,  # gamma1
    0.315,   # gamma3
    0.044,  # gamma4
    0.0, 0.0, 0.0, 0.0  # epsA1, epsB1, epsA2, epsB2
])

data = np.load("pi_bands_from_gpaw.npz")

kpts = data["kpts"]        # shape (Nk, 2)
E_pi = data["energies"]    # shape (Nk, 4)
idx_pi = data["band_indices"]    # π band indices in DFT
EF = data["EF"]

bands = tb_bands(kpts, theta0)
# --- Plot ---
plt.figure(figsize=(6,5))
for i in range(4):
    plt.plot(bands[:,i], lw=2)

# vertical lines & labels
#for p in label_positions:
#    plt.axvline(p, color='gray', linewidth=0.5)

#plt.xticks(label_positions, labels)
plt.ylabel("Energy (eV)")
plt.title("Bilayer Graphene Band Structure (Tight Binding)")
plt.tight_layout()
plt.show()

#print(k_path)