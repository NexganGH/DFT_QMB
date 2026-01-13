from gpaw import GPAW
from external_potentials import register_with_gpaw
register_with_gpaw()

calc = GPAW("../gpw_all/ab_gate_plane_A0.000.gpw", txt=None)
kpt = calc.wfs.kpt_u[0]

print("Wavefunction mode:", calc.wfs.mode)
print("Has psit_nG:", kpt.psit_nG is not None)
if kpt.psit_nG is not None:
    print("psit_nG shape:", kpt.psit_nG.shape)
