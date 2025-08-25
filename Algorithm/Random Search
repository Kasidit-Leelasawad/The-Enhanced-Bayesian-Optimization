###############################################################################
#                      Random Search (Legacy)                                 #
###############################################################################

def random_search_legacy(f, n_params, bounds_rs, n_iters):
    """
    A naive optimization routine that randomly samples the allowed space
    and returns the best value.

    Args:
        f (callable): Objective function to minimize.
        n_params (int): Number of parameters.
        bounds_rs (np.ndarray): (n_params, 2) array with [lower, upper] bounds.
        n_iters (int): Number of random samples/iterations.

    Returns:
        tuple: (best_function_value, best_parameter_array)
    """
    local_x   = np.zeros((n_params, n_iters))  # Points sampled
    local_val = np.zeros(n_iters)            # Function values sampled

    bounds_range = bounds_rs[:, 1] - bounds_rs[:, 0]
    bounds_bias  = bounds_rs[:, 0]

    for sample_i in range(n_iters):
        x_trial = np.random.uniform(0, 1, n_params) * bounds_range + bounds_bias
        local_x[:, sample_i] = x_trial
        local_val[sample_i] = f(x_trial)

    # Choose the best
    min_index = np.argmin(local_val)
    f_best = local_val[min_index]
    x_best = local_x[:, min_index]

    return f_best, x_best
