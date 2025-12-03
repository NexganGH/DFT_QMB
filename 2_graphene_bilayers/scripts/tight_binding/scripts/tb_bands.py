import numpy as np
from ase.geometry import cell_to_cellpar
class TBBandManager:

    def __init__(self, atoms):
        self._atoms = atoms
        a, b, c, alpha, beta, gamma= cell_to_cellpar(atoms.cell)
        self.a = a
        self.a0 = np.sqrt(3) * a


    def _get_deltas(self):
        a = self.a
        return np.array([
            [0,  a/np.sqrt(3)],
            [ a/2, -a/(2*np.sqrt(3))],
            [-a/2, -a/(2*np.sqrt(3))]
        ])

    def f_k(self, kx, ky):
        a, a_0 = self.a, self.a0
        delta = self._get_deltas()
        return np.sum(np.exp(1j * (delta[:,0]*kx + delta[:,1]*ky)))

    def build_kpath(self):
        a, a0 = self.a, self.a0


        b1 = (2 * np.pi / a0) * np.array([1, 1 / np.sqrt(3)])
        b2 = (2 * np.pi / a0) * np.array([1, -1 / np.sqrt(3)])

        # High symmetry points
        Gamma = np.array([0, 0])
        K = (b1 + 2 * b2) / 3
        M = b1 / 2

        # Build the k-path
        N = 200
        k_path = []
        label_positions = []
        labels = [r"$\Gamma$", "K", "M", r"$\Gamma$"]

        # Γ → K
        for t in np.linspace(0, 1, N):
            k_path.append((1 - t) * Gamma + t * K)
        label_positions.append(0)

        # K → M
        offset = len(k_path)
        for t in np.linspace(0, 1, N):
            k_path.append((1 - t) * K + t * M)
        label_positions.append(offset)

        # M → Γ
        offset2 = len(k_path)
        for t in np.linspace(0, 1, N):
            k_path.append((1 - t) * M + t * Gamma)
        label_positions.append(offset2)
        label_positions.append(len(k_path) - 1)

        return np.array(k_path), label_positions, labels

    def H_bilayer(self, kx, ky, params):
        """
        params = (gamma0, gamma1, gamma3, gamma4,
                  epsA1, epsB1, epsA2, epsB2)
        You can drop eps* if you want all zeros.
        """
        (g0, g1, g3, g4, epsA1, epsB1, epsA2, epsB2) = params
        f = self.f_k(kx, ky)
        fc = np.conj(f)

        H = np.zeros((4,4), dtype=complex)
        H[0,0] = epsA1
        H[1,1] = epsB1
        H[2,2] = epsA2
        H[3,3] = epsB2

        H[0,1] = -g0 * f
        H[1,0] = -g0 * fc
        H[2,3] = -g0 * f
        H[3,2] = -g0 * fc

        H[1,2] = g1
        H[2,1] = g1

        H[0,3] = -g3 * fc
        H[3,0] = -g3 * f

        H[0,2] = g4 * f
        H[2,0] = g4 * fc
        H[1,3] = g4 * f
        H[3,1] = g4 * fc

        return H

    def tb_bands(self, kpts, params):
        """
        kpts: array (Nk, 2) with (kx, ky)
        returns: energies (Nk, 4), sorted at each k
        """
        Nk = kpts.shape[0]
        bands = np.zeros((Nk, 4))
        for i, (kx, ky) in enumerate(kpts):
            H = self.H_bilayer(kx, ky, params)
            eigs = np.linalg.eigvalsh(H)
            bands[i,:] = np.sort(np.real(eigs))
        return bands
