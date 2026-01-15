# DFT: Graphene

See [HowToRun.md](HowToRun.md) to setup the environment for DFT.

## 1. Graphene Monolayer
In folder `1_graphene`. All the scripts are available (ordered) in the `scripts` folder.

## 2. Graphene Bilayer
In folder `2_graphene_bilayers`.
- For the normal bilayer (without external field), the main workflow is in `2_graphene_bilayers/1_normal_bilayers/0_full_workflow.ipynb`. The other `.py` are helper files where bigger parts of the code are. The methods are documented.
  - Here you will find the TB bandgap fit and trigonal warping.
- For the tunable bilayer, everything is in the folder `2_graphene_bilayers/2_tunable_bilayer`. Inside this folder: 
  - `0_basic_dft_tunable_bandgap.ipynb`: this is the main reference file to run the biased DFT calculations. 
    - Notice that to run this file, first you must have run `2_graphene_bilayers/1_normal_bilayers/0_full_workflow.ipynb`. That is because it is dependent on the output `2_tunable_bilayer/gpw/ab.gpw`.
    - Notice that `0_basic_dft_tunable_bandgap.ipynb` **must be run** before running all following calculations, as this file will generate several GPW files in the folder `2_tunable_bilayer/gpw` that are used to extract polarization, as well as some `csv` files containing the bandgap.
  - `1_*` to `4_*`: this is the main procedure for the meanfield Hartree Fock calculation.
  - `5_*` to `8_*`: this is the main procedure for the RPA calculation. `9_*` to `10_*` are for plotting purposes.
  - Other files in this folder: utilities such as to define the external potentials (see method documentation).
  - Other folders:
    - `gpw`: contains the GPW files (DFT calculations).
    - `gpw_all`: contains GPW files for the RPA calculation, containing all the plane waves (used to extract the dielectirc function). 
    - `data`: plots and results.
  
## 3. Graphene Nanoribbons
