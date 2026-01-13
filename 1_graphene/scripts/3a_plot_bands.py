import numpy as np
import matplotlib.pyplot as plt

from plotstyle import set_plot_style
from common.mpl_style import set_mpl_style
set_plot_style()
set_mpl_style()


# -----------------------------------------------------
# 1. Load raw band-structure data
# -----------------------------------------------------
data = np.load('../outputs/graphene_bands_raw.npz')

x = data['x']
xticks = data['xticks']
labels = data['labels']
eigs = data['eigs']
fermi = data['fermi']

# -----------------------------------------------------
# 2. Plot bands
# -----------------------------------------------------
plt.figure(figsize=(7, 5))

for band in eigs.T:
    plt.plot(x, band - fermi, 'k-', lw=1)

plt.axhline(0, color='red', linestyle='--', linewidth=0.8)

# Special points (vertical lines)
for xpos in xticks:
    plt.axvline(xpos, color='gray', linewidth=0.5)

plt.xticks(xticks, labels)
plt.ylabel('Energy (eV)')
plt.title('Graphene Band Structure')
plt.tight_layout()

# -----------------------------------------------------
# 3. Save + show
# -----------------------------------------------------
plt.savefig('../outputs/graphene_bandstructure.png', dpi=500)
plt.show()

print("Saved plot to graphene_bandstructure.png")
