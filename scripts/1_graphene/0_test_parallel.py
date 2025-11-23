from gpaw import GPAW

calc = GPAW(mode='fd', kpts=(50,50,1))
print("Calculator created.")
