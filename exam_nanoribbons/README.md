# ZGNRs Code

This project provides a simple user interface to compute the relevant physical quantities in the study of **zigzag graphene nanoribbons (ZGNRs)**.

The code is designed to compare and connect different theoretical descriptions—tight-binding, mean-field Hubbard, and spin-polarized DFT—focusing in particular on **band structures** and **magnetization profiles**.

---

## Project Structure

The project is organized as follows:

project/

│── src/  
│── inputs/  
│── outputs/  
│── run_main.py  
│── README.md  

---

## `src/` Folder

The `src` folder contains the core implementations of the project, including:

- Functions for **non-interacting tight-binding calculations**, used to compute band structures and density of states (DOS).
- The implementation of the **mean-field Hubbard model**, used to compute spin-resolved band structures and on-site magnetization.
- The core routines for **spin-polarized DFT calculations**, which are used to:
  - Recover the full DFT band structure,
  - Isolate the central π bands,
  - Perform a fit to the mean-field Hubbard model,
  - Compute and analyze differences in the on-site magnetization.

---

## `inputs/` Folder

The `inputs` folder contains all files controlling the **input parameters** of the calculations, including:

- Model parameters (e.g. hopping terms, interaction strengths),
- Control flags that determine which calculations are performed,
- Parameters governing numerical accuracy and convergence.

These inputs are read by the execution scripts and can be modified by the user to customize the simulations.

---

## Run Scripts and Outputs

At the same level as the `src` and `inputs` folders, files named `run_*.py` are provided.  
These scripts are the **entry points of the simulations**.

Each run script:
- Reads the relevant input values from the `inputs` folder,
- Calls the appropriate functions implemented in `src`,
- Writes the results to the `outputs` folder in the form of data and visualization files (e.g. `.npz`, `.gpw`, `.png`).

---

Further sections will describe in detail the available input parameters, execution modes, and physical outputs.

## Non-Interacting Tight-Binding Calculations

The non-interacting tight-binding calculations are configured through the file  
`input_non_interacting.py`, located in the `inputs/` directory.

This part of the code is used to compute the **electronic band structure** and **density of states (DOS)** of zigzag graphene nanoribbons (ZGNRs) within a nearest-neighbor, non-interacting tight-binding model.

---

### Input File: `input_non_interacting.py`

All parameters controlling the non-interacting calculations are defined in this file.  
The user must edit this file before running the corresponding simulation.

---

### Run Mode Selection

The code supports two execution modes:

- **Single mode**: calculation for a fixed nanoribbon width \( N \),
- **Sweep mode**: calculation for a range of nanoribbon widths.

The run mode is selected via the variable:

```python
MODE = "single"   # alternatively: "sweep"
```

When `MODE = "single"`:

- The calculation is performed for a single zigzag graphene nanoribbon (ZGNR).
- The ribbon width is specified by `N_SINGLE=...`

When `MODE = "sweep"`:

- The calculation is performed for multiple ribbon widths.
- The ribbon width is swept over the range:
  `
  N_MIN = ...
  N_MAX = ...`

### Tight-Binding Model Parameters

The parameters defining the non-interacting tight-binding Hamiltonian are:

- `t`: nearest-neighbor hopping parameter,
- `a`: lattice constant.

These parameters fully specify the electronic model used to compute the band structure.

### Numerical Accuracy and Convergence Parameters

The numerical resolution of the calculations is controlled by the following parameters:

- `nk`: number of sampled points in the Brillouin zone,
- `eta`: Gaussian smearing parameter used to broaden the density of states (DOS),
- `n_E`: number of energy points used to sample the energy axis in the DOS calculation.

### Running the Non-Interacting Calculation

After setting the desired input parameters, save the input file and run the simulation with:

```bash
python run_non_interacting.py
```

### Output Files

Upon completion, a new output directory named:

```text
postproc_noninteracting/
```
is created at the same level as the `src` and `inputs` folders.

This directory contains:

- Numerical data files storing band structure and DOS results,
- Graphical outputs of the band structure and DOS (e.g. `.png` files).

## Mean-Field Hubbard Calculations

The mean-field Hubbard calculations are configured through the file  
`input_mf.py`, located in the `inputs/` directory.

This part of the code implements a **mean-field Hubbard model** on top of the tight-binding description in order to compute **spin-resolved band structures** and **on-site magnetization** for zigzag graphene nanoribbons.

---

### Run Mode and Interaction Sweep

As in the non-interacting case, the calculation supports the same run modes (`single` and `sweep` in the ribbon width).  
In addition, an extra flag allows one to perform a sweep over the on-site interaction strength.

#### Sweep over the Interaction Strength

The flag:

```python
SWEEP_U = True
```
enables a sweep over the Hubbard interaction parameter \( U \).

When this option is active:

- The ribbon width \( N \) is fixed,
- The interaction strength is swept from:
```
Umin_over_t = ...
Umax_over_t = ...
```
where the interaction is expressed in units of the hopping parameter 
t.This mode is useful to study the evolution of the electronic structure and magnetization as a function of the interaction strength at fixed ribbon width.

### Hubbard Model Parameters

The strength of the on-site interaction can be specified depending on the selected run mode:

- `U_SINGLE`: interaction strength used for a single calculation,
- `U_FIXED_FOR_SWEEP_N`: interaction strength used when sweeping over the ribbon width \( N \).

### Numerical Accuracy and Self-Consistency Parameters

In addition to the numerical parameters used in the non-interacting calculations, the mean-field Hubbard model requires parameters controlling the self-consistent loop:

```python
filling = 1.0      # filling per site (half-filling for graphene: one electron per site)
max_iter = 300     # maximum number of self-consistent iterations
mix = 0.10         # mixing factor: m <- (1 - mix) * m_old + mix * m_new
tol = 1e-5         # convergence threshold on the magnetization
m0 = 0.05          # initial edge magnetization seed
```

### Running the Mean-Field Hubbard Calculations

After setting the desired input parameters in the input file, run the simulation with:

```bash
python run_mf.py
```

Upon completion, a new output directory named:
```text
postproc_mf/
```
will be created at the same level as the other project folders, containing the numerical results of the mean-field calculations.

To generate the plots of the on-site magnetization, run:
```bash
python run_mf_lattice.py
```


## Spin-Polarized DFT Calculations

Spin-polarized DFT calculations are configured through the file  
`inputs_spindft.py`, located in the `inputs/` directory.

This section of the code runs **spin-polarized DFT (GPAW)** for ZGNRs in order to obtain the **full DFT band structure**, and to later extract and analyze the **central π bands** and magnetization.

---

### Run Mode Selection

As in the previous modules, the calculation can be executed either for a single ribbon width or for multiple widths.

```python
# MODE: "single" or "sweep_N"
MODE = "single"

# --- single ---
NY_SINGLE = 2

# --- sweep list ---
NY_LIST = [1, 2, 4, 6, 8]
```
- If `MODE = "single"`, the calculation is performed only for `NY_SINGLE`.
- If `MODE = "sweep_N"`, the sweep is performed over the values listed in `NY_LIST`.


### Geometry Parameters (ASE `graphene_nanoribbon`)

These parameters define the nanoribbon geometry used to build the atomic structure:
```python
M = 1                   # number of unit cells along periodic direction
C_C = 1.42              # C-C distance (Å)
VACUUM = 10.0           # vacuum padding in non-periodic directions (Å)
SATURATED = True        # H-terminated edges
```

- `M`: controls the ribbon length along the periodic direction (supercell repetition).
- `C_C`: sets the carbon–carbon bond length.
- `VACUUM`: adds vacuum spacing to avoid interactions between periodic images.
- `SATURATED`: if `True`, the ribbon edges are hydrogen-terminated.

### DFT Settings (GPAW)

These parameters control the DFT setup and numerical settings:
```python
USE_PW = True           # True: plane waves, False: LCAO
PW_ECUT = 300           # eV (plane-wave cutoff)
LCAO_BASIS = "dzp"      # used only if USE_PW=False

XC = "PBE"
KPTS_Z = 10             # k-points along periodic direction
FERMI_WIDTH = 0.05      # eV (smearing)
```
- `USE_PW`: choose between plane-wave (`True`) and LCAO (`False`) calculations.
- `PW_ECUT`: plane-wave energy cutoff (only if `USE_PW = True`).
- `LCAO_BASIS`: basis set (only if `USE_PW = False`).
- `XC`: exchange-correlation functional.
- `KPTS_Z`: number of k-points along the periodic direction.
- `FERMI_WIDTH`: smearing used for electronic occupations (helps convergence).

### Spin Initialization and Relaxation

These parameters control the initial magnetic seed and optional structural relaxation:
```python
SPIN_SEED = 0.5         # μB initial seed on edges
EDGE_TOL = 0.25         # Å: which C atoms are considered "edge" by x distance

DO_RELAX = False
FMAX = 0.05             # eV/Å for relaxation
MAX_STEPS = 200
```
- `SPIN_SEED`: initial magnetic moment applied to edge atoms to trigger spin polarization.
- `EDGE_TOL`: tolerance (in Å) used to identify which carbon atoms are treated as edge atoms.
- `DO_RELAX`: if `True`, performs geometry relaxation before the final calculation.
- `FMAX`, `MAX_STEPS`: convergence parameters for the relaxation procedure.

### Band Structure Extraction

After the self-consistent density is obtained, bands can be computed non-selfconsistently:
```python
NPOINTS_KPATH = 20      # Γ -> Z sampling
NBANDS_BANDS = 10       # number of bands for band plot/extraction
```
- `NPOINTS_KPATH`: number of k-points along the chosen band path.
- `NBANDS_BANDS`: number of bands included in the band calculation and extraction.

### Run spDFT
To run the spin-polarized DFT calculations, execute:

```bash
python run_spindt.py
```

### Fitting Procedure
In the file
```bash
inputs_tbfit.py
```
the relevant input setting for the fit are set. 
One can set the initial guess for the paramters 
```text
t, U 
```
and then decide the parameters to control the SC routine of the mean-field Hubbrd model and the fallback grid of chosen values of t and U.

To run the Fit, simply execute the code in:
```bash
run_spindft_tbfit.py
```

### Final Post processing 
To compute the final post-processing refinements on the output plots and generated data after the fit.
Run the code in:
```bash
run_spindft_postproc.py
```

