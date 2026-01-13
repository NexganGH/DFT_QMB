# inputs_spindft.py
# Control single run vs sweep and all DFT settings.

# MODE: "single" or "sweep_N"
MODE = "single"

# --- single ---
NY_SINGLE = 2

# --- sweep list ---
NY_LIST = [1, 2, 4, 6, 8]

# ============================================================
# OUTPUT ROOT
# ============================================================
BASE_OUTDIR = "postproc_outputs_spindft"

# ============================================================
# GEOMETRY (ASE graphene_nanoribbon)
# ============================================================
M = 1                   # number of unit cells along periodic direction
C_C = 1.42              # C-C distance (Å)
VACUUM = 10.0           # vacuum padding in non-periodic directions (Å)
SATURATED = True        # H-terminated edges

# ============================================================
# DFT SETTINGS (GPAW)
# ============================================================
USE_PW = True           # True: plane waves, False: LCAO
PW_ECUT = 300           # eV (plane-wave cutoff)
LCAO_BASIS = "dzp"      # used only if USE_PW=False

XC = "PBE"
KPTS_Z = 10 #18             # k-points along periodic direction
FERMI_WIDTH = 0.05      # eV (smearing)

SPIN_SEED = 0.5         # μB initial seed on edges
EDGE_TOL = 0.25         # Å: which C atoms are considered "edge" by x distance

DO_RELAX = False
FMAX = 0.05             # eV/Å for relaxation
MAX_STEPS = 200

# ============================================================
# BANDS (non-selfconsistent bands with fixed density)
# ============================================================
NPOINTS_KPATH = 20 #40      # Γ -> Z sampling
NBANDS_BANDS = 10 #20        # number of bands for band plot/extraction

# π window selection (relative to EF)
E_MIN_PI = -1.5
E_MAX_PI = +1.5
