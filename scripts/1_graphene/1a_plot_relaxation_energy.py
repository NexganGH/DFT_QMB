from ase.io import read
import matplotlib.pyplot as plt

# Read all images from the trajectory
images = read('graphene_relax.traj', index=':')

energies = [img.get_potential_energy() for img in images]

plt.plot(energies, marker='o')
plt.xlabel('Optimization step')
plt.ylabel('Total Energy (eV)')
plt.title('Graphene Relaxation Energy')
plt.grid(True)
plt.show()