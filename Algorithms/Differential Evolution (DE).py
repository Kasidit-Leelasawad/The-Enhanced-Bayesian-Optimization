###############################################################################
#                        Differential Evolution (DE)                          #
###############################################################################

def opt_de(f, x_dim, bounds, iter_tot):
    """
    Minimization using SciPy's Differential Evolution.

    Args:
        f (callable): Objective function.
        x_dim (int): Dimensionality of the search space.
        bounds (np.ndarray): (x_dim, 2) array with lower/upper bounds.
        iter_tot (int): Total budget of function evaluations.

    Returns:
        tuple: (best_parameters, best_function_value)
    """
    # Convert (x_dim, 2) array to list of tuples for DE
    bounds_list = [(float(b[0]), float(b[1])) for b in bounds]

    # Estimate popsize and maxiter
    popsize_ = int(min(100, max(iter_tot * 0.05, 5)))
    maxiter_ = int(iter_tot / popsize_) + 3

    # Differential Evolution
    opt_res = differential_evolution(
        f,
        bounds_list,
        maxiter=maxiter_,
        popsize=popsize_,
    )

    return opt_res.x, opt_res.fun
