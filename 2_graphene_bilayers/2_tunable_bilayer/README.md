#%%
# 2. Tunable bilayer
- `0_basic_dft_tunable_bandgap.ipynb`: all the calculations for the field sweep of the tunable bilayer in DFT.

- `external_potentials.py`: used to apply the external potential to a DFT calculation. Most importantly, it's also used to register the new potential once a file is loaded.

## Hartree Mean field
First, we need to extract the polarization from the DFT calculations. This is then used to fit the term $\Delta - \Delta_{ext}$ and obtain the effective screening. The relation is then solved self-consistently.

- `1_extract_polarization.py` : extract polarization from the DFT calculations. Use `field_sweep_results.csv` from `0_basic_dft_tunable_bandgap.py` and outputs a file `layer_polarisation_vs_field.csv` to be used later.
- `2_analyze_gap_vs_polarization.py`: first analysis to see relation between polarization and bandgap.
- `3_fit_polarization.py`:  fit `layer_polarisation_vs_field.csv` to bandgaps from `bandgaps_KZOOM.csv` to extract $U_H$ and $b_0$. We use the relation $Delta(E_g)$ (see report)
- `4_self_consistent.py`: now we are ready to run the actual SC equations using the parameter $U_H$ and $b_0$ extracted from the fit before.
## RPA
The main approach here is to use dielectric functions from the DFT calculations to find an effective screening, depending on the electric field. 

First, we need to extract the dielectric functions from the DFT calculations. This is done automatically in GPAW. However, it needs the plane waves to compute the correlation function. Because of this, we need to rerun all the DFT calculations also saving the plane waves. After that, we can run the actual RPA calculations and extract the dielectric functions both in `LFC` and `NLFC`. Finally, we apply the same SC procedure that was detailed previously, but now the effective screening depends on the electric field and is computed from the dielectric functions.

- `5_repack_gpw.py`: previous DFT calculations were run without saving all the plane waves, but now we need them for the RPA calculations. Here we rerun the DFT calculations in `mode='all'` to save all of them in `../gpw_all`.
- `6_single_rpa_test.py`: prior test of the RPA calculations.
- `7_run_rpa.py`: Find all gpw files in `../gpw_all` and extract the dielectric functions (both LFC and NLFC), and save them in `epsilon_vs_field.csv`.
- `8_rpa_sc.py`: run the main RPA SC calculations in a kpatch around the K point. The procedure is very similar to the Hartree Mean field approach.
- `9_plot_rpa.py`-`10_plot_rpa_bandgap.py`: plots