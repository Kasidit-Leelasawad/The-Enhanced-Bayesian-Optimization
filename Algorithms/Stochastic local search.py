###############################################################################
#                    Random Search (Stochastic Search)                        #
###############################################################################

def random_search(f, n_params, bounds, n_iters):
    """
    Perform a naive random search over the given parameter space.

    Args:
        f (callable): Objective function to minimize.
        n_params (int): Number of parameters.
        bounds (np.ndarray): (n_params, 2) array with [lower, upper] bounds.
        n_iters (int): Number of samples to evaluate.

    Returns:
        tuple: (best_value, best_params)
    """
    bounds_range = bounds[:, 1] - bounds[:, 0]
    bounds_bias = bounds[:, 0]

    sampled_points = (
        np.random.uniform(0, 1, (n_iters, n_params)) * bounds_range + bounds_bias
    )
    sampled_values = np.array([f(point) for point in sampled_points])

    min_index = np.argmin(sampled_values)
    best_value = sampled_values[min_index]
    best_params = sampled_points[min_index]

    return best_value, best_params

###############################################################################
#                              Ball Sampling                                  #
###############################################################################

def ball_sampling(n_dims, radius):
    """
    Sample randomly within a ball of a given radius in n_dims space.
    """
    u = np.random.normal(0, 1, n_dims)
    norm_u = np.linalg.norm(u)
    r = np.random.uniform() ** (1.0 / n_dims)

    return r * u / norm_u * radius * 2.0

###############################################################################
#                           Log-Uniform Sampling                              #
###############################################################################

def log_uniform_sample(size, min_val=1e-8, max_val=1.0):
    """
    Generate samples uniformly distributed in log-space
    between min_val and max_val.
    """
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    log_samples = np.random.uniform(log_min, log_max, size)

    samples = np.exp(log_samples)
    signs = np.random.choice([-1, 1], size=size)
    samples *= signs

    return samples

###############################################################################
#                        Random Local Search (RLS)                            #
###############################################################################

def random_local_search(f, best_value, best_params, n_samples, n_params, bounds):
    """
    Perform a local random search to refine the solution.

    Args:
        f (callable): Objective function.
        best_value (float): Current best function value.
        best_params (np.ndarray): Current best parameters.
        n_samples (int): Number of local samples to evaluate.
        n_params (int): Dimensionality.
        bounds (np.ndarray): (n_params, 2) array with [lower, upper] bounds.

    Returns:
        tuple: (best_value, best_params, radius_list)
    """
    radius = (bounds[:, 1] - bounds[:, 0]) * 0.5
    samples_per_iteration = n_params*2
    n_iterations = n_samples // samples_per_iteration
    gamma = 0.95
    radius_list = []

    for i_iter in range(n_iterations+1):
        radius_list.append(np.mean(radius))

        if i_iter % 2 == 0:
            # Sample in log space
            sampled_points = np.array([
                best_params + log_uniform_sample(n_params) * radius
                for _ in range(samples_per_iteration)
            ])
        else:
            # Sample in a ball
            sampled_points = np.array([
                best_params + ball_sampling(n_params, radius)
                for _ in range(samples_per_iteration)
            ])

        sampled_points = np.clip(sampled_points, bounds[:, 0], bounds[:, 1])
        sampled_values = np.array([f(point) for point in sampled_points])

        min_index = np.argmin(sampled_values)
        local_best_value = sampled_values[min_index]
        local_best_params = sampled_points[min_index]

        if local_best_value < best_value:
            best_value = local_best_value
            best_params = local_best_params
        else:
            radius *= gamma

    return best_value, best_params, radius_list

###############################################################################
#                  Stochastic Search (Global + Local)                         #
###############################################################################

def SS_alg(f, n_params, bounds, n_iters):
    """
    Perform a stochastic search combining global random search and local random search.

    Args:
        f (callable): Objective function to minimize.
        n_params (int): Number of parameters.
        bounds (np.ndarray): (n_params, 2) array with [lower, upper] bounds.
        n_iters (int): Total number of function evaluations.

    Returns:
        tuple: (best_params, best_value, radius_list)
    """
    n_random_search_iters = int(n_iters * 0.1) + 1
    n_local_search_iters = n_iters - n_random_search_iters

    best_value, best_params = random_search(f, n_params, bounds, n_random_search_iters)
    best_value, best_params, radius_list = random_local_search(
        f, best_value, best_params, n_local_search_iters, n_params, bounds
    )

    return best_params, best_value, radius_list
