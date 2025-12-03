import numpy as np
from ase.geometry import cell_to_cellpar
import numpy as np
import plotly.graph_objects as go
from gpaw import GPAW, restart

class TBBandManager:
    """
    Manager for Tight-Binding (TB) band structure analysis.

    This class handles the construction and analysis of tight-binding band
    structures for materials. It provides utilities to calculate various
    properties, such as k-paths, band energies, and Hamiltonians for bilayer
    systems, as well as methods for 3D band visualization using interactive
    techniques.

    :ivar a: Lattice constant extracted from the atomic cell.
    :type a: float
    :ivar a0: Adjusted lattice constant scaled by √3.
    :type a0: float
    """
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

    def create_3d_plot(self, theta, kpts=None):
        atoms, calc = restart('../gpw/ab.gpw')

        # ======================================================
        # 0. USER INPUTS (you MUST define these before running)
        # ======================================================
        # kpts_tb  = TB k-path: shape (N1, 2)
        #kpts_tb, label_positions, labels = bands_manager.build_kpath()
       # kpts_ase = dft_kpts
        # kpts_ase = ASE k-path: shape (N2, 2)
        # tb_bands = function that takes (Nk, 2) array of kpts
        #            and returns energies shape (nbands, Nk)

        # Example:
        # E = tb_bands(kpts)   # returns (nbands, Nk)

        # ======================================================
        # 1. Build a 2D grid over kx, ky (same grid as your code)
        # ======================================================
        a0 = self.a0
        Nk = 60
        kx_vals = np.linspace(-2 * np.pi / a0, 2 * np.pi / a0, Nk)
        ky_vals = np.linspace(-2 * np.pi / a0, 2 * np.pi / a0, Nk)
        KX, KY = np.meshgrid(kx_vals, ky_vals)

        # Flatten to shape (Nk^2, 2)
        k_grid = np.c_[KX.ravel(), KY.ravel()]

        # ======================================================
        # 2. Compute TB bands on the grid using your tb_bands()
        # ======================================================

        Eall = self.tb_bands(k_grid, theta)  # shape (nbands, Nk*Nk)
        Nk2, nbands = Eall.shape  # Nk2 = Nk*Nk

        bands = np.zeros((Nk, Nk, nbands))
        for n in range(nbands):
            # take column n (all k-points for band n) and reshape
            bands[:, :, n] = Eall[:, n].reshape(Nk, Nk)

        # ======================================================
        # 3. Create interactive 3D Plotly figure
        # ======================================================

        fig = go.Figure()

        # --- plot band surfaces ---
        for n in range(nbands):
            fig.add_trace(go.Surface(
                x=KX, y=KY, z=bands[:, :, n],
                colorscale="Viridis",
                opacity=0.65,
                showscale=False,
                name=f"Band {n + 1}"
            ))

        # ======================================================
        # 4. Add TB k-path (your manual path)
        # ======================================================

        # compute energies along TB path
        # E_tb_path = self.tb_bands(kpts_tb, theta0)  # shape (nbands, Nt)
        # # choose which band to plot (0 = lowest)
        # band_idx = 0
        # fig.add_trace(go.Scatter3d(
        #     x=kpts_tb[:, 0],
        #     y=kpts_tb[:, 1],
        #     z=E_tb_path[:, band_idx],  # pick one band
        #     mode='lines',
        #     line=dict(color='red', width=6),
        #     name='TB path'
        # ))

        # ======================================================
        # 5. Add ASE k-path (after convert → cartesian)
        # ======================================================
        #
        # E_ase_path = bands_manager.tb_bands(kpts_ase, theta0)
        # fig.add_trace(go.Scatter3d(
        #     x=kpts_ase[:, 0],
        #     y=kpts_ase[:, 1],
        #     z=E_ase_path[:, band_idx],
        #     mode='lines',
        #     line=dict(color='blue', width=6),
        #     name='ASE path'
        # ))

        # ======================================================
        # 6. Final layout
        # ======================================================

        fig.update_layout(
            title="3D Band Surfaces + TB & ASE k-Paths",
            scene=dict(
                xaxis_title="kx",
                yaxis_title="ky",
                zaxis_title="Energy (eV)",
                xaxis=dict(backgroundcolor="black"),
                yaxis=dict(backgroundcolor="black"),
                zaxis=dict(backgroundcolor="black"),
            ),
            width=950,
            height=800
        )

        # ======================================================
        # 7. Save figure
        # ======================================================

        output_path = "../data/3d_bands.html"
        fig.write_html(output_path)

        print(f'File saved in {output_path}')
