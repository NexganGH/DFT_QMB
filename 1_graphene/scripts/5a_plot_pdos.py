# plot_pdos_pi_sigma.py
import os
import numpy as np
import matplotlib.pyplot as plt
from common.mpl_style import set_mpl_style
set_mpl_style()
outdir = '../outputs/'

# Load data
e_s, pdos_s       = np.loadtxt(os.path.join(outdir, 'graphene_pdos_s.dat')).T
e_p, pdos_p_tot   = np.loadtxt(os.path.join(outdir, 'graphene_pdos_p_total.dat')).T
e_pi, pdos_pi     = np.loadtxt(os.path.join(outdir, 'graphene_pdos_pi.dat')).T
e_sig, pdos_sigma = np.loadtxt(os.path.join(outdir, 'graphene_pdos_sigma.dat')).T

plt.figure(figsize=(7,5))

plt.plot(e_pi,  pdos_pi,     label=r'$\pi$ (p$_z$)',        linewidth=2)
plt.plot(e_sig, pdos_sigma,  label=r'$\sigma$ (p$_x$+p$_y$)', linewidth=2)
plt.plot(e_s,   pdos_s,      label='s', linestyle='--', linewidth=1)

plt.axvline(0.0, linestyle='--', color='grey')
plt.xlim(-10, 10)
plt.ylim(bottom=0)

plt.xlabel(r'$E - E_F$ (eV)')
plt.ylabel('PDOS (states/eV)')
plt.title('Graphene PDOS: π and σ')
plt.legend()
plt.tight_layout()

outfile = os.path.join(outdir, 'graphene_pdos_pi_sigma.png')
plt.savefig(outfile, dpi=500)
plt.show()

print("Saved plot:", outfile)
