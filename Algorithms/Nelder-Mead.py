###############################################################################
#                             Nelder-Mead                                    #
###############################################################################
from scipy.optimize import minimize as scipy_minimize
def opt_nelder_mead(f, x_dim, bounds, iter_tot):
    """
    Minimization using the Nelder-Mead method from SciPy.

    Args:
        f (callable): Objective function.
        x_dim (int): Dimensionality of the search space.
        bounds (np.ndarray): (x_dim, 2) array with lower/upper bounds.
        iter_tot (int): Total budget of function evaluations.

    Returns:
        tuple: (best_parameters, best_function_value)
    """
    # Attempt to find a good starting point via random search
    n_rs = int(min(100, max(iter_tot * 0.2, 5)))
    f_best, x_best = random_search_legacy(f, x_dim, bounds, n_rs)

    # Remaining evaluations after random search
    remaining_evals = iter_tot - n_rs

    # Nelder-Mead minimization
    opt_res = scipy_minimize(
        f, x_best,
        method="Nelder-Mead",
        options={"maxfev": remaining_evals}
    )

    return opt_res.x, opt_res.fun
