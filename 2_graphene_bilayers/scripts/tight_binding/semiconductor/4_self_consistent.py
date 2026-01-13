import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
OUT_DIR = "../gpw"
GAP_CSV = os.path.join(OUT_DIR, "bandgaps_KZOOM.csv")
POL_CSV = os.path.join(OUT_DIR, "layer_polarisation_vs_field.csv")
OUT_MODEL_CSV = os.path.join(OUT_DIR, "self_consistent_tb_results.csv")

# Fields to evaluate (if you want to use the DFT sweep fields)
EZ_MAX = 0.005
ROUND_EZ_DECIMALS = 3

# Geometry / units
d_interlayer_A = 3.35  # Å
# Convention used here: Delta_ext [eV] = Ez [V/Å] * d [Å]

# ------------------------------------------------------------
# Hartree parameters (YOU SET THESE)
# ------------------------------------------------------------
U_H_eV_per_e = 0.257   # <-- set from your manual linear fit (slope), units eV / electron
b0_eV = 0.133168          # <-- optional intercept in Delta - Delta_ext = U_H*P + b0

# ------------------------------------------------------------
# Tight-binding parameters (minimal AB bilayer, pi-only)
# ------------------------------------------------------------
a_cc = 1.42          # Å  C-C bond
gamma0_eV = 2.687     # eV  intralayer nearest-neighbour hopping (|gamma0| ~ 3 eV)
gamma1_eV = 0.262     # eV  interlayer dimer hopping (B1 <-> A2)

# k-mesh for BZ integration (increase for smoother results)
NK1 = 80
NK2 = 80

# Self-consistency solver settings
DELTA_MAX_eV = 1.5
BISECTION_ITERS = 60
SC_TOL_eV = 1e-6


# ============================================================
# CSV loaders
# ============================================================
def load_bandgaps(path):
    Ez, Eg = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                Ez.append(float(row["Ez"]))
                Eg.append(float(row["bandgap_eV"]))
            except Exception:
                continue
    return np.array(Ez), np.array(Eg)


def load_polarisation(path):
    Ez, P = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                Ez.append(float(row["Ez"]))
                P.append(float(row["layer_polarisation_electrons"]))
            except Exception:
                continue
    return np.array(Ez), np.array(P)


def merge_by_rounded_Ez(Ez1, A1, Ez2, A2, decimals=3):
    k1 = np.round(Ez1, decimals)
    k2 = np.round(Ez2, decimals)
    m1 = {float(e): a for e, a in zip(k1, A1)}
    m2 = {float(e): a for e, a in zip(k2, A2)}
    keys = sorted(set(m1) & set(m2))
    Ez = np.array(keys)
    A1m = np.array([m1[k] for k in keys])
    A2m = np.array([m2[k] for k in keys])
    return Ez, A1m, A2m


# ============================================================
# Graphene lattice / reciprocal lattice
# ============================================================
def graphene_lattice_vectors(a=2.46):
    """
    Graphene primitive lattice vectors (Å).
    a ~ 2.46 Å is the lattice constant (not a_cc).
    """
    a1 = np.array([0.5 * a, 0.5 * np.sqrt(3) * a])
    a2 = np.array([-0.5 * a, 0.5 * np.sqrt(3) * a])
    return a1, a2


def reciprocal_vectors(a1, a2):
    """
    Return b1, b2 such that a_i · b_j = 2π δ_ij.
    """
    A = np.stack([a1, a2], axis=1)  # columns
    B = 2.0 * np.pi * np.linalg.inv(A).T
    b1 = B[:, 0]
    b2 = B[:, 1]
    return b1, b2


def k_mesh_bz(N1, N2, b1, b2):
    """
    Uniform mesh over the primitive reciprocal parallelogram.
    k = u*b1 + v*b2 with u,v in [0,1).
    """
    us = (np.arange(N1) + 0.5) / N1
    vs = (np.arange(N2) + 0.5) / N2
    kpts = []
    for u in us:
        for v in vs:
            kpts.append(u * b1 + v * b2)
    return np.array(kpts)  # shape (N1*N2, 2)


# ============================================================
# TB Hamiltonian: AB bilayer, gamma0-gamma1, layer potential +/-Delta/2
# Basis: (A1, B1, A2, B2)
# ============================================================
def f_k(kx, ky, a_cc=a_cc):
    """
    Graphene nearest-neighbour structure factor f(k) = sum exp(i k·delta_j).
    delta vectors in Å.
    """
    d1 = np.array([0.0, a_cc])
    d2 = np.array([0.5 * np.sqrt(3) * a_cc, -0.5 * a_cc])
    d3 = np.array([-0.5 * np.sqrt(3) * a_cc, -0.5 * a_cc])
    k = np.array([kx, ky])
    return np.exp(1j * np.dot(k, d1)) + np.exp(1j * np.dot(k, d2)) + np.exp(1j * np.dot(k, d3))


def H_ab_bilayer(kx, ky, Delta_eV,
                 gamma0=gamma0_eV, gamma1=gamma1_eV):
    """
    4x4 TB Hamiltonian for AB bilayer graphene (minimal).
    Layer potentials: +Delta/2 on layer 1, -Delta/2 on layer 2.
    """
    fk = f_k(kx, ky)
    t = -gamma0 * fk

    D1 = +0.5 * Delta_eV
    D2 = -0.5 * Delta_eV

    H = np.zeros((4, 4), dtype=complex)

    # Onsite layer potentials
    H[0, 0] = D1  # A1
    H[1, 1] = D1  # B1
    H[2, 2] = D2  # A2
    H[3, 3] = D2  # B2

    # Intralayer hoppings
    H[0, 1] = t
    H[1, 0] = np.conjugate(t)
    H[2, 3] = t
    H[3, 2] = np.conjugate(t)

    # Interlayer dimer coupling (AB stacking): B1 <-> A2
    H[1, 2] = gamma1
    H[2, 1] = gamma1

    return H


# ============================================================
# Compute P(Delta) and Eg(Delta) from TB on a k-mesh
# ============================================================
def tb_polarisation_and_gap(Delta_eV, kpts):
    """
    Returns:
      P(Delta): induced layer polarisation in electrons per unit cell
               (layer2 - layer1), with spin degeneracy included (x2).
      Eg(Delta): band gap (eV) from TB: min conduction - max valence.
    """
    # Accumulate layer polarisation contributions
    pol_sum = 0.0

    # Track band edges
    val_max = -1e9
    cond_min = +1e9

    for kx, ky in kpts:
        H = H_ab_bilayer(kx, ky, Delta_eV)
        evals, evecs = np.linalg.eigh(H)  # evals ascending

        # At charge neutrality: occupy 2 lowest bands (per spin)
        occ_bands = [0, 1]

        for b in occ_bands:
            vec = evecs[:, b]
            w_layer1 = (np.abs(vec[0])**2 + np.abs(vec[1])**2)
            w_layer2 = (np.abs(vec[2])**2 + np.abs(vec[3])**2)
            pol_sum += (w_layer2 - w_layer1)

        # Band edges
        val_max = max(val_max, evals[1])   # top of valence (2nd band)
        cond_min = min(cond_min, evals[2]) # bottom of conduction (3rd band)

    Nk = len(kpts)
    # Average over k; multiply by spin degeneracy 2
    P = 2.0 * (pol_sum / Nk)

    Eg = max(0.0, cond_min - val_max)
    return float(P), float(Eg)


# ============================================================
# Self-consistent solver: Delta = Delta_ext + U_H * P(Delta) + b0
# ============================================================
def Delta_ext_from_Ez(Ez):
    return float(Ez) * d_interlayer_A


def sc_residual(Delta, Delta_ext, U_H, b0, kpts):
    P, _ = tb_polarisation_and_gap(Delta, kpts)
    return Delta - (Delta_ext + b0 + U_H * P)


def solve_self_consistent_delta(Delta_ext, U_H, b0, kpts,
                                Delta_max=DELTA_MAX_eV,
                                iters=BISECTION_ITERS,
                                tol=SC_TOL_eV):
    """
    Solve F(Delta)=0 by bisection on [0, Delta_max].
    Assumes Delta >= 0 for Ez >= 0 (you can extend sign symmetry if needed).
    """
    lo, hi = 0.0, float(Delta_max)
    Flo = sc_residual(lo, Delta_ext, U_H, b0, kpts)
    Fhi = sc_residual(hi, Delta_ext, U_H, b0, kpts)

    # If not bracketed, expand hi a bit (simple)
    if Flo * Fhi > 0:
        for _ in range(8):
            hi *= 1.5
            Fhi = sc_residual(hi, Delta_ext, U_H, b0, kpts)
            if Flo * Fhi <= 0:
                break

    if Flo * Fhi > 0:
        # No root found in range; return best-effort fixed-point iterate
        Delta = Delta_ext
        for _ in range(50):
            P, _ = tb_polarisation_and_gap(Delta, kpts)
            Delta_new = Delta_ext + b0 + U_H * P
            if abs(Delta_new - Delta) < tol:
                return float(Delta_new)
            Delta = Delta_new
        return float(Delta)

    # Bisection
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Fm = sc_residual(mid, Delta_ext, U_H, b0, kpts)

        if abs(Fm) < tol:
            return float(mid)

        if Flo * Fm <= 0:
            hi = mid
            Fhi = Fm
        else:
            lo = mid
            Flo = Fm

    return float(0.5 * (lo + hi))


# ============================================================
# Main
# ============================================================
def main():
    # Prepare k-mesh
    # Graphene lattice constant a ~ sqrt(3)*a_cc
    a_lat = np.sqrt(3.0) * a_cc
    a1, a2 = graphene_lattice_vectors(a=a_lat)
    b1, b2 = reciprocal_vectors(a1, a2)
    kpts = k_mesh_bz(NK1, NK2, b1, b2)

    # Load DFT data (optional)
    Ez_gap, Eg_dft = load_bandgaps(GAP_CSV)
    Ez_pol, P_dft = load_polarisation(POL_CSV)

    # Determine the fields we will run: prefer fields that exist in DFT gap CSV
    if Ez_gap.size > 0:
        Ez_list = np.unique(np.round(Ez_gap, ROUND_EZ_DECIMALS))
    elif Ez_pol.size > 0:
        Ez_list = np.unique(np.round(Ez_pol, ROUND_EZ_DECIMALS))
    else:
        # fallback
        Ez_list = np.round(np.linspace(0.0, EZ_MAX, 11), ROUND_EZ_DECIMALS)

    Ez_list = Ez_list[Ez_list <= EZ_MAX]
    Ez_list = np.sort(Ez_list)

    # For comparisons, merge DFT P and DFT gap on common Ez (if both exist)
    Ez_common = np.array([])
    Eg_common = np.array([])
    P_common = np.array([])
    if Ez_gap.size > 0 and Ez_pol.size > 0:
        Ez_common, Eg_common, P_common = merge_by_rounded_Ez(
            Ez_gap, Eg_dft, Ez_pol, P_dft, decimals=ROUND_EZ_DECIMALS
        )
        m = Ez_common <= EZ_MAX
        Ez_common, Eg_common, P_common = Ez_common[m], Eg_common[m], P_common[m]

    # Run self-consistency
    results = []
    for Ez in Ez_list:
        Delta_ext = Delta_ext_from_Ez(Ez)
        Delta_sc = solve_self_consistent_delta(Delta_ext, U_H_eV_per_e, b0_eV, kpts)

        P_sc, Eg_sc = tb_polarisation_and_gap(Delta_sc, kpts)

        results.append((Ez, Delta_ext, Delta_sc, P_sc, Eg_sc))
        print(f"[SC] Ez={Ez:.3f}  Delta_ext={Delta_ext:.4f} eV  Delta_sc={Delta_sc:.4f} eV  "
              f"P={P_sc:.4f} e/cell  Eg={Eg_sc:.4f} eV")

    # Save results
    with open(OUT_MODEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ez", "Delta_ext_eV", "Delta_sc_eV", "P_sc_e_per_cell", "Eg_sc_eV"])
        for row in results:
            w.writerow(row)

    print(f"\nSaved model results to: {OUT_MODEL_CSV}")

    # Plot: Eg(Ez) model vs DFT
    Ez_m = np.array([r[0] for r in results])
    Eg_m = np.array([r[4] for r in results])
    P_m = np.array([r[3] for r in results])
    Delta_m = np.array([r[2] for r in results])

    plt.figure()
    plt.plot(Ez_m, Eg_m, "o-", label="TB+Hartree self-consistent Eg")
    if Ez_common.size > 0:
        plt.plot(Ez_common, Eg_common, "o", label="DFT Eg (KZOOM)")
    plt.xlabel("E_z")
    plt.ylabel("Band gap (eV)")
    plt.title("Self-consistent TB+Hartree gap vs field")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, P_m, "o-", label="TB+Hartree P(Delta_sc)")
    if Ez_common.size > 0:
        plt.plot(Ez_common, P_common, "o", label="DFT P_ind")
    plt.xlabel("E_z")
    plt.ylabel("Layer polarisation P (e / cell)")
    plt.title("Self-consistent TB+Hartree polarisation vs field")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ez_m, Delta_m, "o-", label="Delta_sc")
    plt.plot(Ez_m, np.array([r[1] for r in results]), "--", label="Delta_ext")
    plt.xlabel("E_z")
    plt.ylabel("Delta (eV)")
    plt.title("Internal asymmetry: self-consistent vs bare")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
