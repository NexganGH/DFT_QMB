from ase.spectrum.band_structure import BandStructure
from ase.io.jsonio import read_json
import matplotlib.pyplot as plt

bs: BandStructure = read_json('bandstructure_AB.json')

print(bs.energies)
bs.plot()
plt.show()