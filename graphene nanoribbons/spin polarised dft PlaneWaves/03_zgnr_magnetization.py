# 03_zgnr_magnetization.py

from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt
import config_zgnr as cfg


# ------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------

def choose_transverse_coord(pos):
    """Choose transverse axis = axis with larger span among x,y."""
    span_x = pos[:, 0].max() - pos[:, 0].min()
    span_y = pos[:, 1].max() - pos[:, 1].min()

    if span_x >= span_y:
        return pos[:, 0], 'x'
    else:
        return pos[:, 1], 'y'


def choose_periodic_coord(pos):
    """Periodic direction = axis with largest span."""
    spans = [
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min()
    ]
    labels = ['x', 'y', 'z']
    idx = int(np.argmax(spans))
    return pos[:, idx], labels[idx]


def magnetic_profile_bins(t, magmoms, symbols, nbins, only_carbon=True):
    """Compute coarse-grained m(t) across the ribbon."""
    t = np.array(t)
    magmoms = np.array(magmoms)
    symbols = np.array(symbols)

    if only_carbon:
        mask = (symbols == 'C')
        t = t[mask]
        magmoms = magmoms[mask]

    tmin = t.min()
    tmax = t.max()

    bins = np.linspace(tmin, tmax, nbins + 1)
    indices = np.digitize(t, bins) - 1  # 0..nbins-1

    m_per_bin = np.zeros(nbins)
    count_per_bin = np.zeros(nbins, dtype=int)

    for idx, m in zip(indices, magmoms):
        if 0 <= idx < nbins:
            m_per_bin[idx] += m
            count_per_bin[idx] += 1

    t_centers = 0.5 * (bins[:-1] + bins[1:])
    return t_centers, m_per_bin, count_per_bin


def group_C_rows_by_transverse(tC, C_indices, tol):
    """Group C atoms into rows by transverse coord."""
    order = np.argsort(tC)
    sorted_indices = C_indices[order]

    rows = []
    current_row = [sorted_indices[0]]

    for a, b in zip(sorted_indices[:-1], sorted_indices[1:]):
        if abs(tC[b] - tC[a]) < tol:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]

    rows.append(current_row)
    return rows


def strands_from_rows(rows):
    """Pair rows → strands."""
    n_rows = len(rows)
    if n_rows % 2 != 0:
        print("WARNING: odd number of C rows.")

    n_strands = n_rows // 2
    strands = []

    for m in range(n_strands):
        group = rows[2*m] + rows[2*m + 1]
        strands.append(group)

    if n_rows % 2 != 0:
        strands.append(rows[-1])

    return strands


def AB_mags(strands, z_per, magmoms):
    """Extract m_A and m_B per strand."""
    m_indices = []
    mA = []
    mB = []
    listA = []
    listB = []

    for m_idx, strand in enumerate(strands):
        row = np.array(strand, dtype=int)
        order_z = np.argsort(z_per[row])
        row_sorted = row[order_z]

        A_atoms = row_sorted[0::2]
        B_atoms = row_sorted[1::2]

        mA_row = magmoms[A_atoms].mean() if len(A_atoms) > 0 else 0.0
        mB_row = magmoms[B_atoms].mean() if len(B_atoms) > 0 else 0.0

        m_indices.append(m_idx)
        mA.append(mA_row)
        mB.append(mB_row)

        listA.append(A_atoms)
        listB.append(B_atoms)

        print(
            f"strand m={m_idx}: A={A_atoms.tolist()},  mA={mA_row:.3f}; "
            f"B={B_atoms.tolist()},  mB={mB_row:.3f}"
        )

    return (np.array(m_indices),
            np.array(mA),
            np.array(mB),
            listA,
            listB)


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    calc = GPAW(cfg.gpw_file)
    atoms = calc.get_atoms()

    pos = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    magmoms = np.array(calc.get_magnetic_moments())

    # detect transverse, periodic direction
    t, trans_label = choose_transverse_coord(pos)
    z_per, per_label = choose_periodic_coord(pos)

    print("Transverse axis:", trans_label)
    print("Periodic axis:", per_label)

    # m(t)
    t_centers, m_per_bin, count_per_bin = (
        magnetic_profile_bins(
            t, magmoms, symbols,
            nbins=cfg.Ny_bins,
            only_carbon=True
        )
    )

    # C rows and strands
    C_indices = np.where(symbols == 'C')[0]
    tC = t[C_indices]

    rows = group_C_rows_by_transverse(tC, C_indices, tol=cfg.row_tol)
    print("C row count:", len(rows))

    strands = strands_from_rows(rows)
    print("Strand count:", len(strands))

    m_indices, mA, mB, listA, listB = AB_mags(strands, z_per, magmoms)

    # --- Plot m(t) ---
    plt.figure(figsize=(6,4))
    plt.axhline(0.0, color='k', ls='--')
    plt.plot(t_centers, m_per_bin, 'o-')
    plt.xlabel(f"{trans_label} (Å)")
    plt.ylabel("m(t) (μB)")
    plt.title("Transverse magnetization profile")
    plt.tight_layout()
    plt.savefig(cfg.mag_profile_png, dpi=300)
    plt.close()

    # --- Plot A/B per strand ---
    plt.figure(figsize=(6,4))
    plt.plot(m_indices, mA, 'o-', label='A')
    plt.plot(m_indices, mB, 's-', label='B')
    plt.axhline(0.0, color='k', ls='--')
    plt.xlabel("Strand index m")
    plt.ylabel("Magnetization (μB)")
    plt.title("A / B magnetization per strand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.mag_AB_png, dpi=300)
    plt.close()

    # --- Save data ---
    np.savez(
        cfg.mag_profile_npz,
        t_centers=t_centers,
        m_per_bin=m_per_bin,
        count_per_bin=count_per_bin,
        trans_label=trans_label
    )

    np.savez(
        cfg.mag_AB_npz,
        m_indices=m_indices,
        mA=mA,
        mB=mB,
        listA=listA,
        listB=listB,
        trans_label=trans_label,
        per_label=per_label
    )

    print(f"Saved: {cfg.mag_profile_npz}")
    print(f"Saved: {cfg.mag_AB_npz}")


if __name__ == "__main__":
    main()
