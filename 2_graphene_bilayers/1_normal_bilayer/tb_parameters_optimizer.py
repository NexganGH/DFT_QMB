from scipy.optimize import least_squares
import numpy as np

def optimize(bands_manager, theta0, kpts, E_pi, min_x, max_x):

    def cost_function(params, kpts, E_dft):
        E_tb = bands_manager.tb_bands(kpts, params)
        return (E_tb - E_dft).ravel()

    # k-path distance
    dk = np.sqrt(np.sum(np.diff(kpts, axis=0)**2, axis=1))
    x = np.concatenate([[0], np.cumsum(dk)])

    mask = (x > min_x) & (x < max_x)

    res = least_squares(
        cost_function,
        theta0,
        args=(kpts[mask], E_pi[mask]),
        max_nfev=200,
        verbose=2
    )

    theta_fit = res.x

    # ---- ERROR ESTIMATION ----
    n_data = res.fun.size
    n_params = theta_fit.size

    residual_variance = np.sum(res.fun**2) / (n_data - n_params)

    J = res.jac
    cov = residual_variance * np.linalg.inv(J.T @ J)

    theta_err = np.sqrt(np.diag(cov))

    return theta_fit, theta_err