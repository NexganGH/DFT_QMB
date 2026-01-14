# spindft_magnetization_core.py
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW

def choose_transverse_coord(pos):
    span_x = pos[:, 0].max() - pos[:, 0].min()
    span_y = pos[:, 1].max() - pos[:, 1].min()
    if span_x >= span_y:
        return pos[:, 0], "x"
    return pos[:, 1], "y"

def choose_periodic_coord(pos):
    spans = [
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min(),
    ]
    labels = ["x", "y", "z"]
    idx = int(np.argmax(spans))
    return pos[:, idx], labels[idx]

def group_C_rows_by_transverse(tC, C_indices, tol):
    order = np.argsort(tC)
    sorted_indices = C_indices[order]
    rows = []
    current = [sorted_indices[0]]
    for a, b in zip(sorted_indices[:-1], sorted_indices[1:]):
        if abs(tC[b] - tC[a]) < tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    return rows

def strands_from_rows(rows):
    n_rows = len(rows)
    n_strands = n_rows // 2
    strands = []
    for m in range(n_strands):
        strands.append(rows[2*m] + rows[2*m + 1])
    if n_rows % 2 != 0:
        strands.append(rows[-1])
    return strands

def AB_mags(strands, z_per, magmoms):
    mA, mB = [], []
    for strand in strands:
        row = np.array(strand, dtype=int)
        order = np.argsort(z_per[row])
        row_sorted = row[order]
        A_atoms = row_sorted[0::2]
        B_atoms = row_sorted[1::2]
        mA.append(magmoms[A_atoms].mean() if len(A_atoms) else 0.0)
        mB.append(magmoms[B_atoms].mean() if len(B_atoms) else 0.0)
    return np.array(mA, float), np.array(mB, float)

def extract_magnetization(
    gpw_path: str,
    Ny_bins: int,
    row_tol: float,
    mag_profile_png: str,
    mag_AB_png: str,
    mag_profile_npz: str,
    mag_AB_npz: str,
):
    calc = GPAW(gpw_path)
    atoms = calc.get_atoms()
    pos = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    magmoms = np.array(calc.get_magnetic_moments())

    t, trans_label = choose_transverse_coord(pos)
    z_per, per_label = choose_periodic_coord(pos)

    # coarse profile (bins)
    maskC = (symbols == "C")
    tC_all = t[maskC]
    mC_all = magmoms[maskC]
    bins = np.linspace(tC_all.min(), tC_all.max(), Ny_bins + 1)
    idx = np.digitize(tC_all, bins) - 1
    m_per_bin = np.zeros(Ny_bins)
    count = np.zeros(Ny_bins, int)
    for ii, mm in zip(idx, mC_all):
        if 0 <= ii < Ny_bins:
            m_per_bin[ii] += mm
            count[ii] += 1
    t_centers = 0.5*(bins[:-1] + bins[1:])

    # A/B per strand
    C_indices = np.where(symbols == "C")[0]
    tC = t[C_indices]
    rows = group_C_rows_by_transverse(tC, C_indices, tol=row_tol)
    strands = strands_from_rows(rows)
    mA, mB = AB_mags(strands, z_per, magmoms)
    m_idx = np.arange(len(mA))

    # plots (basic; you’ll replace style later if you want)
    plt.figure(figsize=(6,4))
    plt.axhline(0.0, color="0.4", ls="--")
    plt.plot(t_centers, m_per_bin, "o-")
    plt.xlabel(f"{trans_label} (Å)")
    plt.ylabel("magnetization (μB)")
    plt.tight_layout()
    if mag_profile_png:
     plt.savefig(mag_profile_png, dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.axhline(0.0, color="0.4", ls="--")
    plt.plot(m_idx, mA, "o-", label="A")
    plt.plot(m_idx, mB, "s-", label="B")
    plt.xlabel("strand index")
    plt.ylabel("magnetization (μB)")
    plt.legend()
    plt.tight_layout()
    if mag_profile_png:
     plt.savefig(mag_AB_png, dpi=300)
    plt.close()

    np.savez(mag_profile_npz,
             t_centers=t_centers, m_per_bin=m_per_bin, count=count,
             trans_label=trans_label)

    np.savez(mag_AB_npz,
             m_indices=m_idx, mA=mA, mB=mB,
             trans_label=trans_label, per_label=per_label)

    return mA, mB
