###############################################################################
#                                DIRECT                                       #
###############################################################################

def opt_direct(f, x_dim, bounds, iter_tot):
    """
    Minimization using the 'direct' algorithm (if available via SciPy).

    Args:
        f (callable): Objective function.
        x_dim (int): Dimensionality of the search space.
        bounds (np.ndarray): (x_dim, 2) array with lower/upper bounds.
        iter_tot (int): Total budget of function evaluations.

    Returns:
        tuple: (best_parameters, best_function_value)
    """
    bounds_list = [(float(b[0]), float(b[1])) for b in bounds]
    opt_res = direct(f, bounds_list, maxfun=iter_tot)
    return opt_res.x, opt_res.fun
