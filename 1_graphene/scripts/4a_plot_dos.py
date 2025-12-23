# plot_dos.py
import matplotlib.pyplot as plt
import numpy as np
import os
import plotstyle
from common.mpl_style import set_mpl_style
plotstyle.set_plot_style()
set_mpl_style()

outdir = '../outputs/'

# ----------------------------------------------------
# Load DOS data
# ----------------------------------------------------
data = np.loadtxt(outdir + 'graphene_dos.dat')
energies = data[:, 0]
dos = data[:, 1]

# ----------------------------------------------------
# Plot DOS
# ----------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(energies, dos)

plt.axvline(0.0, linestyle='--', color='grey')
plt.xlabel('$E - E_F$ (eV)')
plt.ylabel('DOS')
plt.title('Graphene DOS')
plt.xlim(-10, 10)

plt.tight_layout()
plt.savefig(outdir + 'graphene_dos.png', dpi=300)
plt.show()

print("Plot saved to:", outdir + "graphene_dos.png")
