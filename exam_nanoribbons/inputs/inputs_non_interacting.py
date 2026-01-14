# inputs_non_interacting.py

#to run a single case of N=N_singel choose the lable "single"
#to run a sweep of cases between N_MIN and N_MAX choose the lable "sweep"

#Note that if you choose "single" the control variables N_MIN and N_MAX are irrelevant.
#The same is true for N_SINGLE if the choosen flag is "sweep"

MODE = "single"   # "single" or "sweep"

N_SINGLE = 4

N_MIN = 1
N_MAX = 3

#choice parameters model
t = -2.6 #Hopping term
a = 1.0 #Lattice constant

#choice of accuracy parameters
nk = 1000 #number of sampled points in the Brellouin zone
eta = 0.05 #Smearing of the Gaussians used to plot DOS
n_E = 1000 #

mu = 0.0 #

base_dir = "postproc_outputs_noninteracting"
