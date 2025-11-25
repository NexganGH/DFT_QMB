# plot_dos.py
import matplotlib.pyplot as plt
import numpy as np
import os
import plotstyle

plotstyle.set_plot_style()

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
plt.xlabel('Energy - $E_F$ (eV)')
plt.ylabel('DOS (states/eV)')
plt.title('Graphene DOS (PBE)')
plt.xlim(-10, 10)

plt.tight_layout()
plt.savefig(outdir + 'graphene_dos.png', dpi=300)
plt.show()

print("Plot saved to:", outdir + "graphene_dos.png")
