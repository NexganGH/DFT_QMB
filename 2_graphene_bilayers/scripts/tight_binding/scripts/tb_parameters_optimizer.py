from scipy.optimize import least_squares
import numpy as np

def optimize(bands_manager, theta0, kpts, E_pi, min_x, max_x):
    def cost_function(params, kpts, E_dft):
        """Return flatten residual vector (least_squares format)."""
       # global cost_history

        E_tb = bands_manager.tb_bands(kpts, params)  # (Nk, 4)
        residual = (E_tb - E_dft).ravel()

       # dk = np.sqrt(np.sum(np.diff(kpts, axis=0) ** 2, axis=1))
       # x = np.concatenate([[0], np.cumsum(dk)])

        cost = np.sum(residual ** 2)
       # cost_history.append(cost)

        return residual

    dk = np.sqrt(np.sum(np.diff(kpts, axis=0) ** 2, axis=1))
    x = np.concatenate([[0], np.cumsum(dk)])

    filter = (x > min_x) & (x < max_x)

    res = least_squares(cost_function,
                        theta0,
                        args=(kpts[filter], E_pi[filter]),
                        max_nfev=200,
                        verbose=2)

    theta_fit = res.x
    return theta_fit