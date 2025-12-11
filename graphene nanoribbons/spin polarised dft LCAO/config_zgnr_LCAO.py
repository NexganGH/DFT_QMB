# config_zgnr_LCAO.py

# --- Geometria ---
Ny = 4          # numero di catene zigzag (larghezza)
M = 1           # numero di celle primitive lungo il ribbon
C_C = 1.42      # distanza C-C (Å)
vacuum_x = 10.0 # vuoto lungo x (trasversale)
vacuum_y = 10.0 # vuoto lungo y (trasversale)

# --- Profilo di magnetizzazione ---
Ny_bins = Ny        # numero di bin trasversali ~ numero di strands
row_tol = 0.25      # tolleranza (Å) per raggruppare atomi in "rows" di C
edge_tol = 0.3      # tolleranza (Å) per identificare edge C in x

# --- DFT / GPAW ---
xc = 'PBE'
kpts_z = 40         # k-points lungo direzione periodica (che sarà z)
fermi_width = 0.01  # Fermi-Dirac smearing (eV)
spin_seed = 0.30    # momento magnetico iniziale in modulo sugli edge

# Scegli modalità di calcolo:
use_pw = False      # True -> PW, False -> LCAO

pw_ecut = 200       # eV, se use_pw = True
lcao_basis = 'dzp'  # se use_pw = False

# --- Relax ---
do_relax = True
fmax = 0.05         # eV/Å
max_steps = 100

# --- Band structure ---
nbands_bands = 40   # numero di bande per calc bands
npoints_kpath = 200 # punti lungo Γ–Z
E_min_pi = -3.0     # finestra energia pi (plot)
E_max_pi = +3.0

# --- Nomi file di output ---
geom_traj = f"zgnr_Ny{Ny}_M{M}_zperiodic.traj"
geom_xyz  = f"zgnr_Ny{Ny}_M{M}_zperiodic.xyz"
gpw_file  = f"zgnr_Ny{Ny}_M{M}_scf.gpw"

mag_atoms_npz     = f"zgnr_Ny{Ny}_M{M}_mag_atoms.npz"
mag_profile_npz   = f"zgnr_Ny{Ny}_M{M}_mag_profile.npz"
mag_profile_png   = f"zgnr_Ny{Ny}_M{M}_mag_profile.png"
mag_AB_npz        = f"zgnr_Ny{Ny}_M{M}_mag_AB_strands.npz"
mag_AB_png        = f"zgnr_Ny{Ny}_M{M}_mag_AB_strands.png"

bands_pi_npz      = f"zgnr_Ny{Ny}_M{M}_bands_pi.npz"
bands_pi_png      = f"zgnr_Ny{Ny}_M{M}_bands_pi.png"
bands_full_png    = f"zgnr_Ny{Ny}_M{M}_bands_full.png"

# --- Fit TB+MF ai bands centrali DFT ---
tb_fit_npz       = f"zgnr_Ny{Ny}_M{M}_tb_fit.npz"
tb_fit_bands_png = f"zgnr_Ny{Ny}_M{M}_tb_fit_bands.png"
tb_fit_mag_png   = f"zgnr_Ny{Ny}_M{M}_tb_fit_mag.png"
