# inputs_tbfit.py
# TB mean-field fit settings (fit t and U to DFT central π bands)

# Use the SAME output root as DFT
BASE_OUTDIR = "postproc_outputs_spindft"

# TB solver parameters
NK_TB_FACTOR = 2          # Nk_tb = NK_TB_FACTOR * Nk_dft
FILLING = 1.0             # half filling (1 e per site)
MAX_ITER = 300
MIX = 0.10
TOL = 1e-5

# Fit method
USE_SCIPY_IF_AVAILABLE = True

# Initial guess (eV)
T0 = 2.7
U0 = 2.0

# Fallback grid search if SciPy not found
T_GRID = [2.0, 2.3, 2.6, 2.9, 3.2]
U_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
