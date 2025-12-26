import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================
a_cc = 1.42            # Å
gamma0 = 2.687         # eV
gamma1 = 0.262         # eV

# Test Delta values
DELTA_LIST = np.linspace(0.0, 0.15, 8)

# k meshes
NK_BZ = 60             # full BZ mesh (NxN)
NK_RADIAL = 25
NK_ANGULAR = 60
K_RADIUS = 0.08        # 1/Å

# ============================================================
# Lattice
# ============================================================
def graphene_reciprocal_vectors(a_cc):
    a = np.sqrt(3.0) * a_cc
    a1 = np.array([ a/2,  np.sqrt(3)*a/2])
    a2 = np.array([-a/2,  np.sqrt(3)*a/2])
    A = np.stack([a1, a2], axis=1)
    B = 2*np.pi*np.linalg.inv(A).T
    return B[:,0], B[:,1]


def K_point(b1, b2):
    return (b1 + 2*b2) / 3.0


def kmesh_full_bz(b1, b2, N):
    ks = []
    us = (np.arange(N) + 0.5) / N
    vs = (np.arange(N) + 0.5) / N
    for u in us:
        for v in vs:
            ks.append(u*b1 + v*b2)
    return np.array(ks)


def kpatch(K, radius, nr, na):
    ks = []
    for i in range(nr):
        r = radius * (i + 0.5) / nr
        for j in range(na):
            theta = 2*np.pi * j / na
            ks.append(K + r*np.array([np.cos(theta), np.sin(theta)]))
    return np.array(ks)


# ============================================================
# TB Hamiltonian
# ============================================================
def f_k(kx, ky):
    d1 = np.array([0.0, a_cc])
    d2 = np.array([ np.sqrt(3)*a_cc/2, -a_cc/2])
    d3 = np.array([-np.sqrt(3)*a_cc/2, -a_cc/2])
    return (
        np.exp(1j*(kx*d1[0] + ky*d1[1])) +
        np.exp(1j*(kx*d2[0] + ky*d2[1])) +
        np.exp(1j*(kx*d3[0] + ky*d3[1]))
    )


def H_ab(kx, ky, Delta):
    fk = f_k(kx, ky)
    t = -gamma0 * fk
    D1, D2 = +0.5*Delta, -0.5*Delta

    H = np.zeros((4,4), dtype=complex)
    H[0,0] = H[1,1] = D1
    H[2,2] = H[3,3] = D2

    H[0,1] = t
    H[1,0] = np.conj(t)
    H[2,3] = t
    H[3,2] = np.conj(t)

    H[1,2] = gamma1
    H[2,1] = gamma1
    return H


# ============================================================
# Polarisation
# ============================================================
def tb_polarisation(Delta, kpts):
    pol = 0.0
    for kx, ky in kpts:
        evals, vecs = np.linalg.eigh(H_ab(kx, ky, Delta))
        for b in [0,1]:  # occupied bands
            v = vecs[:,b]
            w1 = np.abs(v[0])**2 + np.abs(v[1])**2
            w2 = np.abs(v[2])**2 + np.abs(v[3])**2
            pol += (w2 - w1)
    return 2.0 * pol / len(kpts)   # spin factor


# ============================================================
# Main diagnostic
# ============================================================
def main():
    b1, b2 = graphene_reciprocal_vectors(a_cc)
    K = K_point(b1, b2)

    k_bz = kmesh_full_bz(b1, b2, NK_BZ)
    k_patch = kpatch(K, K_RADIUS, NK_RADIAL, NK_ANGULAR)

    # BZ / patch area ratio
    A_patch = np.pi * K_RADIUS**2
    A_bz = np.linalg.norm(np.cross(b1, b2))
    weight = A_patch / A_bz

    print(f"BZ area       = {A_bz:.4f}")
    print(f"Patch area    = {A_patch:.4f}")
    print(f"Patch/BZ frac = {weight:.4e}\n")

    P_bz = []
    P_patch = []
    P_patch_corr = []

    for Delta in DELTA_LIST:
        p_bz = tb_polarisation(Delta, k_bz)
        p_p  = tb_polarisation(Delta, k_patch)
        p_pc = weight * p_p

        P_bz.append(p_bz)
        P_patch.append(p_p)
        P_patch_corr.append(p_pc)

        print(f"Δ={Delta:.3f} eV | "
              f"P_BZ={p_bz:.4e} | "
              f"P_patch={p_p:.4e} | "
              f"P_patch_corr={p_pc:.4e}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plt.figure()
    plt.plot(DELTA_LIST, P_bz, "o-", label="Full BZ (reference)")
    plt.plot(DELTA_LIST, P_patch, "o--", label="K-patch (raw)")
    plt.plot(DELTA_LIST, P_patch_corr, "o-", label="K-patch (area-corrected)")
    plt.xlabel("Δ (eV)")
    plt.ylabel("P (e / cell)")
    plt.title("TB layer polarisation diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
