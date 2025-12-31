import sys
import numpy as np
from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from external_potentials import register_with_gpaw

register_with_gpaw()

gpw_file = sys.argv[1]
out_file = sys.argv[2]

print("=" * 60)
print(f"[RPA] Loading ground state: {gpw_file}")

calc = GPAW(gpw_file, txt=None)

nbands = calc.get_number_of_bands()
print(f"[RPA] Number of bands: {nbands}")

df = DielectricFunction(
    calc,
    frequencies=[0.0],
    eta=0.01,
    nbands=nbands,
    hilbert=False
)

Nk = 18
qpts = [
    [1.0 / Nk, 0.0, 0.0],
    [2.0 / Nk, 0.0, 0.0],
    [3.0 / Nk, 0.0, 0.0],
]



eps_q = []

for q in qpts:
    print(f"[RPA] Computing ε(q) at q = {q}")
    eps = df.get_dielectric_function(q_c=q)[0, 0]
    print(f"[RPA]   ε = {eps:.6f}")
    eps_q.append(eps)

eps_q = np.array(eps_q)

print(f"[RPA] Saving ε(q) to: {out_file}")
np.save(out_file, {
    "qpts": np.array(qpts),
    "epsilon": eps_q
})

print("[RPA] Done.")
