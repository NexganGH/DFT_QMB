import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Fit interpolator / summary script
#
# Run this from inside:
#   .../zgnr_sweep_results/fit interpolator.py
#
# Folder structure assumed:
#   zgnr_sweep_results/
#       Ny1/
#           central band selection, fitting and band magnetisation graphs/
#               zgnr_Ny1_M1_tb_fit.npz
#       Ny2/
#           central band selection, fitting and band magnetisation graphs/
#               zgnr_Ny2_M1_tb_fit.npz
#       Ny4/ ...
#       Ny6/ ...
#       Ny8/ ...
#
# Each *_tb_fit.npz contains (at least):
#   Ny, t_fit, U_fit, rms_bands, ...
# ============================================================


def find_Ny_folders(root):
    """Return sorted list of (Ny_int, folder_name) found under root."""
    Ny_list = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name.startswith("Ny"):
            try:
                Ny_val = int(name[2:])
            except ValueError:
                continue
            Ny_list.append((Ny_val, name))
    Ny_list.sort(key=lambda x: x[0])
    return Ny_list


def main():
    this_dir = os.path.dirname(__file__)  # zgnr_sweep_results
    inner_dir = "central band selection, fitting and band magnetisation graphs"

    Ny_folders = find_Ny_folders(this_dir)
    if not Ny_folders:
        raise RuntimeError("No Ny* folders found next to this script.")

    Ny_vals = []
    t_vals = []
    U_vals = []
    rms_vals = []

    print("Collecting fit data from:")
    for Ny, folder_name in Ny_folders:
        fit_dir = os.path.join(this_dir, folder_name, inner_dir)
        fit_file = os.path.join(
            fit_dir, f"zgnr_Ny{Ny}_M1_tb_fit.npz"
        )

        if not os.path.exists(fit_file):
            print(f"  [WARNING] Missing file for Ny={Ny}: {fit_file}")
            continue

        data = np.load(fit_file)
        print(f"  Ny={Ny}: loaded {fit_file}")

        # required keys
        t_fit = float(data["t_fit"])
        U_fit = float(data["U_fit"])
        rms = float(data["rms_bands"])

        Ny_vals.append(Ny)
        t_vals.append(t_fit)
        U_vals.append(U_fit)
        rms_vals.append(rms)

    if not Ny_vals:
        raise RuntimeError("No fit data loaded. Check file names/paths.")

    Ny_vals = np.array(Ny_vals, dtype=int)
    t_vals = np.array(t_vals, dtype=float)
    U_vals = np.array(U_vals, dtype=float)
    rms_vals = np.array(rms_vals, dtype=float)

    print("\nSummary table (Ny, t_fit, U_fit, rms_bands):")
    for Ny, t, U, rms in zip(Ny_vals, t_vals, U_vals, rms_vals):
        print(f"  Ny={Ny:2d}  t={t:7.4f} eV  U={U:7.4f} eV  rms={rms:7.4f} eV")

    # ========================================================
    # 1. Plot t(Ny) with error bars (using rms_bands as error)
    # ========================================================

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.errorbar(
        Ny_vals,
        t_vals,
        yerr=rms_vals,
        fmt="o-",
        capsize=4,
        linewidth=1.8,
        markersize=6,
    )
    ax1.set_xlabel("Number of strands Ny")
    ax1.set_ylabel(r"$t_{\mathrm{fit}}$ (eV)")
    ax1.set_title(r"Fitted hopping $t$ vs ribbon width")
    ax1.grid(True, linestyle=":", alpha=0.4)
    fig1.tight_layout()
    fig1.savefig("t_fit_vs_Ny.png", dpi=300)
    print("Saved plot: t_fit_vs_Ny.png")

    # ========================================================
    # 2. Plot U(Ny) with error bars (same rms as error)
    # ========================================================

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.errorbar(
        Ny_vals,
        U_vals,
        yerr=rms_vals,
        fmt="s-",
        capsize=4,
        linewidth=1.8,
        markersize=6,
    )
    ax2.set_xlabel("Number of strands Ny")
    ax2.set_ylabel(r"$U_{\mathrm{fit}}$ (eV)")
    ax2.set_title(r"Fitted Hubbard $U$ vs ribbon width")
    ax2.grid(True, linestyle=":", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig("U_fit_vs_Ny.png", dpi=300)
    print("Saved plot: U_fit_vs_Ny.png")

    # ========================================================
    # 3. Save summary data for post-processing
    # ========================================================

    out_npz = "zgnr_tb_fit_summary.npz"
    np.savez(
        out_npz,
        Ny=Ny_vals,
        t_fit=t_vals,
        U_fit=U_vals,
        rms_bands=rms_vals,
    )
    print(f"Saved summary data to: {out_npz}")


if __name__ == "__main__":
    main()
