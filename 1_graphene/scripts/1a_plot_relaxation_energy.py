from ase.io import read
import matplotlib.pyplot as plt

images = read('../outputs/graphene_relax.traj', index=':')
energies = [img.get_potential_energy() for img in images]

print(energies)
plt.plot(energies, marker='o')
plt.xlabel('Optimization step')
plt.ylabel('Total Energy (eV)')
plt.title('Graphene Relaxation Energy')
plt.grid(True)

plt.ticklabel_format(style='plain', axis='y')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.gca().yaxis.set_tick_params(pad=8)  # <-- ADD THIS

plt.show()
