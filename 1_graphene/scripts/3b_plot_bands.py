import numpy as np
import matplotlib.pyplot as plt

from plotstyle import set_plot_style
from common.mpl_style import set_mpl_style
set_plot_style()
set_mpl_style()

# -----------------------------------------------------
# 1. Load data
# -----------------------------------------------------
data = np.load('../outputs/graphene_bands_raw.npz')

x = data['x']
xticks = data['xticks']
labels = data['labels']
eigs = data['eigs']
fermi = data['fermi']

# -----------------------------------------------------
# 2. Locate K point
# -----------------------------------------------------
K_index = labels.tolist().index('K')
xK = xticks[K_index]

# zoom window (in k-path units)
dk = 0.15 * (xticks[1] - xticks[0])

xmask = (x >= xK - dk) & (x <= xK + dk)

# -----------------------------------------------------
# 3. Plot zoomed bands
# -----------------------------------------------------
plt.figure(figsize=(7, 5))

for band in eigs.T:
    plt.plot(x[xmask], band[xmask] - fermi, 'k-', lw=1)

plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.axvline(xK, color='gray', linestyle='--', linewidth=0.8)

plt.xlim(xK - dk, xK + dk)
plt.ylim(-1.0, 1.0)  # adjust if needed

plt.xticks([xK], ['K'])
plt.ylabel('Energy (eV)')
plt.title('Graphene Band Structure (Zoom at K)')
plt.tight_layout()

# -----------------------------------------------------
# 4. Save + show
# -----------------------------------------------------
plt.savefig('../outputs/graphene_bandstructure_Kzoom.png', dpi=500)
plt.show()

print("Saved plot to graphene_bandstructure_Kzoom.png")
