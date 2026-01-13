# inputs_mf_lattice.py

# Where to scan for run_summary.npz
BASE_DIR = "postproc_outputs_mf"

# Only make lattice plots for N in this range AND N <= LATTICE_MAX_N
N_MIN = 1
N_MAX = 50
LATTICE_MAX_N = 8

# If edge colors look inverted, toggle this
FLIP_WIDTH_ORDER = True

# Geometry of schematic
N_REPEAT = 12
DX_STEP = 0.90
Y_ZIG = 0.35
DY_STRAND = 1.60

# Output names (inside each run's magnetization folder)
OUT_AUTO_NAME = "mag_lattice_auto.png"
OUT_FIXED_NAME = "mag_lattice_fixed.png"

# Where to collect frames (under BASE_DIR)
ALL_DIR_AUTO = "_ALL/lattice_mag_auto"
ALL_DIR_FIXED = "_ALL/lattice_mag_fixed"

# Figure style
FIGSIZE = (12.0, 5.8)
DPI = 250

# Fixed-box padding (for GIF stability)
FIX_PAD_X = 0.80
FIX_PAD_Y = 0.90
