# inputs_mf.py

# ===========================================================
# What to run?
# ===========================================================

#for an individual run for a given N=N_SINGLE choose "single" flag
#far a sweep run across N_MIN and N_MAX choose "sweep_N" flag
#For a sweep fun on the interaction parameter U (keeping N fixed) choose "sweep_U" flag

MODE = "single"     # "single" | "sweep_N" | "sweep_U"

# --- single mode ---
N_SINGLE = 6
U_SINGLE = 2.7      # eV

# --- sweep over N (fixed U) ---
N_MIN = 2
N_MAX = 5
U_FIXED_FOR_SWEEP_N = 2.7  # eV

# --- sweep over U/t (fixed N) ---
N_FIXED_FOR_SWEEP_U = 6
Umin_over_t = 1.0
Umax_over_t = 3.0
dU_over_t   = 0.10

# ===========================================================
# Model parameters
# ===========================================================
t = 2.4      # eV, hopping coeff
a = 1.0      #lattice constant

# ===========================================================
# Solver accuracy and stability parameters
# ===========================================================

Nk = 400    #number of k points in the 1D Brellouin zone
filling = 1.0   #filling of the site, half filling is the case for graphene, i.e. one electron per site
max_iter = 300  #maximum number of itaration of the Self-consisten loop
mix = 0.10    #mixing factor of the old and new magnetisation at each step: m <- (1 - mix) * m_old + mix * m_new
tol = 1e-5     #threshold on the magnetisation for teh Self-consistent loop
m0 = 0.05            # initial edge magnetization seed
verbose = True

# ===========================================================
# DOS settings
# ===========================================================
dos_nE = 600
dos_eta = 0.05

# ===========================================================
# Output base folder (created in project root)
# ===========================================================
base_dir = "postproc_outputs_mf"
