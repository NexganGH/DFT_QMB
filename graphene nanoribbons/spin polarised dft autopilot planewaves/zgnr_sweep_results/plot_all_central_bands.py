import os
import numpy as np
import matplotlib.pyplot as plt


def find_Ny_folders(root):
    """Return sorted list of (Ny_int, folder_name) under root."""
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
    # This file lives in .../zgnr_sweep_results/
    this_dir = os.path.dirname(__file__)
    inner_dir = "central band selection, fitting and band magnetisation graphs"
    Ny_folders = find_Ny_folders(this_dir)

    if not Ny_folders:
        raise RuntimeError("No Ny* folders found next to this script.")

    all_Ny = []
    all_E_val = []
    all_E_cond = []
    k_ref = None

    print("Collecting central π bands from:")
    for Ny, folder_name in Ny_folders:
        npz_path = os.path.join(
            this_dir,
            folder_name,
            inner_dir,
            "zgnr_pi_smooth_from_Z.npz",
        )

        if not os.path.exists(npz_path):
            print(f"  [WARNING] Missing file for Ny={Ny}: {npz_path}")
            continue

        data = np.load(npz_path)
        print(f"  Ny={Ny}: loaded {npz_path}")

        k = data["k_dimless"]
        E_val = data["E_val"]
        E_cond = data["E_cond"]

        if k_ref is None:
            k_ref = k
        else:
            if len(k) != len(k_ref) or not np.allclose(k, k_ref):
                raise ValueError(
                    f"k-grid mismatch for Ny={Ny}. "
                    "All ribbons must share the same k_dimless."
                )

        all_Ny.append(Ny)
        all_E_val.append(E_val)
        all_E_cond.append(E_cond)

    if not all_Ny:
        raise RuntimeError("No central-band data loaded; nothing to plot.")

    all_Ny = np.array(all_Ny, dtype=int)
    order = np.argsort(all_Ny)
    all_Ny = all_Ny[order]
    all_E_val = [all_E_val[i] for i in order]
    all_E_cond = [all_E_cond[i] for i in order]

    # ============================================================
    # 1. Plot all central π bands together
    # ============================================================

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # nice: one color per Ny, conduction=solid, valence=dashed of same color
    for idx, Ny in enumerate(all_Ny):
        color = f"C{idx % 10}"
        E_val = all_E_val[idx]
        E_cond = all_E_cond[idx]

        ax.plot(
            k_ref / np.pi,
            E_cond,
            lw=2.0,
            color=color,
            label=fr"$N_y={Ny}$ (cond.)",
        )
        ax.plot(
            k_ref / np.pi,
            E_val,
            lw=1.6,
            ls="--",
            color=color,
            alpha=0.8,
        )

    ax.axhline(0.0, ls="--", lw=0.8, color="k", alpha=0.6)

    ax.set_xlabel(r"$k a / \pi$", fontsize=13)
    ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=13)
    ax.set_title(
        r"Central $\pi$ bands for spin-polarised ZGNRs (PW-DFT)",
        fontsize=14,
    )

    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        fontsize=9,
        ncol=2,
        frameon=True,
        framealpha=0.9,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig("central_pi_bands_all_Ny.png", dpi=300)
    print("Saved combined plot: central_pi_bands_all_Ny.png")

    # ============================================================
    # 2. Save combined data for post-processing
    # ============================================================

    out_npz = "central_pi_bands_all_Ny.npz"
    all_E_val_arr = np.vstack(all_E_val)   # shape (N_Ny, Nk)
    all_E_cond_arr = np.vstack(all_E_cond)

    np.savez(
        out_npz,
        Ny=all_Ny,
        k_dimless=k_ref,
        E_val=all_E_val_arr,
        E_cond=all_E_cond_arr,
    )
    print(f"Saved combined data to: {out_npz}")


if __name__ == "__main__":
    main()
