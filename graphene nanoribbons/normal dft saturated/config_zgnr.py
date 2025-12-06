# config_zgnr.py

import os
import sys

# ---------- Basic ZGNR options ----------

N_ZIGZAG = 4                # width: ZGNR-N
LENGTH_REPEATS = 1           # number of primitive cells along x, in reality it is z
VACUUM = 12.0                # Å in non-periodic directions

LABEL_BASE = f"zgnr{N_ZIGZAG}_nonmag"

# ---------- DFT options ----------

KPTS_1D_RELAX = 20           # SCF k-points along x
ECUT = 400                   # eV, plane-wave cutoff
FMAX = 0.02                  # eV/Å, relaxation force criterion

NK_PATH = 250                # k-points along Γ→X
NBANDS = 40                  # number of bands in band structure
ETA_DOS = 0.05               # eV, Gaussian broadening for DOS

# ---------- TB import (from sibling folder) ----------

# Folder name and module name of your TB code
TB_FOLDER_NAME = "non interacting"        # <-- your folder with the space
TB_MODULE_NAME = "dos_non_interacting"    # <-- your file name without .py

# Paths
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TB_DIR = os.path.join(PROJECT_ROOT, TB_FOLDER_NAME)


def get_tb_module():
    """
    Import and return the TB module that contains H_zgnr_k.
    """
    if TB_DIR not in sys.path:
        sys.path.append(TB_DIR)
    tb_mod = __import__(TB_MODULE_NAME)
    return tb_mod
