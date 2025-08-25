###############################################################################
#                     Particle Swarm Optimization (PSO)                       #
###############################################################################

def domain_reduction_v2(x_dic, n_particles):
    """
    Reduces search space by taking the min/max among all best positions.
    """
    new_domain = x_dic['0']['best_position']
    for i_x in range(1, n_particles):
        new_domain = np.vstack((new_domain, x_dic[str(i_x)]['best_position']))

    max_bound = np.max(new_domain, axis=0)
    min_bound = np.min(new_domain, axis=0)

    return np.vstack((min_bound, max_bound)).T

def sample_uniform_params_v2(n_particles_, bounds_range, bounds_bias):
    """
    Uniform sampling of parameters in a given bounds_range and bounds_bias.
    """
    return (
        np.random.uniform(0, 1, (n_particles_, bounds_range.shape[0]))
        * bounds_range + bounds_bias
    )

def sample_uniform_params_log_v2(n_particles_, bounds_range, bounds_bias):
    """
    Uniform sampling in logarithmic space.
    """
    log_bounds_range = np.log(bounds_range)
    rand_particles_log = np.random.uniform(
        0, 1, (n_particles_, bounds_range.shape[0])
    ) * log_bounds_range

    return np.exp(rand_particles_log) + bounds_bias

def calculate_v_log_v2(w_k, c1, c2, v_p, x_p, x_local_best, x_global_best):
    """
    Calculates particle velocity in log-space for exploration-exploitation.
    """
    diff_local = x_local_best - x_p
    diff_global = x_global_best - x_p

    abs_diff_local = np.abs(diff_local)
    abs_diff_global = np.abs(diff_global)

    log_abs_diff_local = np.log(abs_diff_local + 1e-10)
    log_abs_diff_global = np.log(abs_diff_global + 1e-10)

    rand_local = np.random.uniform(0, 1, x_p.shape)
    rand_global = np.random.uniform(0, 1, x_p.shape)

    scaled_log_local = rand_local * log_abs_diff_local
    scaled_log_global = rand_global * log_abs_diff_global

    scaled_local = np.sign(diff_local) * np.exp(scaled_log_local)
    scaled_global = np.sign(diff_global) * np.exp(scaled_log_global)

    v_new = w_k * v_p + c1 * scaled_local + c2 * scaled_global
    return v_new

def calculate_v_v2(w_k, c1, c2, v_p, x_p, x_local_best, x_global_best):
    """
    Calculates particle velocity using the standard PSO formula.
    """
    inertia = w_k * v_p
    local_comp = c1 * np.random.uniform(0, 1, x_p.shape) * (x_local_best - x_p)
    global_comp = c2 * np.random.uniform(0, 1, x_p.shape) * (x_global_best - x_p)

    return inertia + local_comp + global_comp

def pso_v2(n_particles, evals, bounds, func, x_good=None, f_good=None):
    """
    Basic PSO algorithm to find the minimum of a given function.
    """
    if x_good is None:
        x_good = []
    if f_good is None:
        f_good = np.inf

    eval_count = 0
    best_reward = np.inf

    c1, c2 = 2.8, 1.3
    c3 = c1 + c2
    # Velocity weighting factor (inertia)
    w_k = 2.0 / abs(2.0 - c3 - np.sqrt(c3**2 - 4 * c3))

    bounds_range = bounds[:, 1] - bounds[:, 0]
    bounds_bias = bounds[:, 0]
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    v_lb = -abs(bounds_range) * 0.75
    v_ub = abs(bounds_range) * 0.75

    data_pso = {'R_list': np.zeros(evals)}
    particle_dic = {}

    # If we have a good starting point
    if len(x_good) > 0:
        n_p = n_particles - 1
        particle_dic[str(n_p)] = {
            'particle_x': x_good.copy(),
            'best_position': x_good.copy(),
            'best_obj': f_good,
            'vel': sample_uniform_params_v2(1, v_ub - v_lb, v_lb)[0]
        }
        best_reward = f_good
        best_particle = x_good.copy()
        eval_count += 1
        nn = 1
    else:
        nn = 0
        best_particle = None

    # Initialize other particles
    for particle_i in range(n_particles - nn):
        if np.random.uniform(0, 1) > 0.5:
            sample_uniform_ = sample_uniform_params_v2
            sample_velocity_ = calculate_v_v2
        else:
            sample_uniform_ = sample_uniform_params_log_v2
            sample_velocity_ = calculate_v_log_v2

        part_x = sample_uniform_(1, bounds_range, bounds_bias)[0]
        particle_dic[str(particle_i)] = {
            'particle_x': part_x.copy(),
            'best_position': part_x.copy(),
            'best_obj': np.inf,
            'vel': sample_uniform_(1, v_ub - v_lb, v_lb)[0]
        }

        data_pso['R_list'][eval_count] = func(particle_dic[str(particle_i)]['particle_x'])
        particle_dic[str(particle_i)]['best_obj'] = data_pso['R_list'][eval_count]

        if data_pso['R_list'][eval_count] < best_reward:
            best_reward = data_pso['R_list'][eval_count]
            best_particle = particle_dic[str(particle_i)]['particle_x'].copy()

        eval_count += 1

    # Main PSO loop
    while eval_count < evals:
        for p_i in range(n_particles):
            if np.random.uniform(0, 1) > 0.5:
                sample_velocity_ = calculate_v_v2
            else:
                sample_velocity_ = calculate_v_log_v2

            p_dict = particle_dic[str(p_i)]
            p_dict['vel'] = sample_velocity_(
                w_k, c1, c2,
                p_dict['vel'],
                p_dict['particle_x'],
                p_dict['best_position'],
                best_particle
            )

            # Update position and clip to bounds
            p_dict['particle_x'] = p_dict['particle_x'] + p_dict['vel']
            p_dict['particle_x'] = np.clip(
                p_dict['particle_x'],
                lower_bounds,
                upper_bounds
            )

            # Evaluate
            data_pso['R_list'][eval_count] = func(p_dict['particle_x'])

            if data_pso['R_list'][eval_count] < p_dict['best_obj']:
                p_dict['best_obj'] = data_pso['R_list'][eval_count]
                p_dict['best_position'] = p_dict['particle_x'].copy()

                if p_dict['best_obj'] < best_reward:
                    best_reward = p_dict['best_obj']
                    best_particle = p_dict['particle_x'].copy()

            eval_count += 1
            if eval_count >= evals:
                break

    return best_particle, best_reward, data_pso, particle_dic

###############################################################################
#      PSO with Domain Reduction + Local Search (PSO_red_v2)                  #
###############################################################################

def pso_red_v2(n_particles, evals, bounds, func):
    """
    A repeated PSO approach where after each cycle the domain is reduced.
    Finally, a local search is performed for further refinement.
    """
    iter_per_cycle = int(evals / 4)

    # First cycle
    best_x, best_f, data_pso, x_dic = pso_v2(n_particles, iter_per_cycle, bounds, func)

    # Two additional cycles of domain reduction + PSO
    for _ in range(2):
        new_b = domain_reduction_v2(x_dic, n_particles)
        best_x, best_f, data_pso, x_dic = pso_v2(
            n_particles, iter_per_cycle, new_b, func, x_good=best_x, f_good=best_f
        )

    # Local search
    n_x    = best_x.shape[0]
    new_b  = domain_reduction_v2(x_dic, n_particles)
    iter_t = int(iter_per_cycle * (n_particles / n_x + 1))

    best_x, best_f, _ = random_local_search(func, best_f, best_x, iter_t, n_x, new_b)

    return best_x, best_f, data_pso, x_dic

###############################################################################
#        Wrapper: Repeated PSO with Local Search (pso_red_f_v2)              #
###############################################################################

def pso_red_f_v2(f, x_dim, bounds, iter_tot):
    """
    High-level function that runs repeated PSO with local search.

    Args:
        f (callable): Objective function.
        x_dim (int): Dimensionality of the search space.
        bounds (np.ndarray): (x_dim, 2) array with [lower, upper] bounds.
        iter_tot (int): Total number of function evaluations.

    Returns:
        tuple: (best_x, best_f) where best_x are the best parameters found,
               and best_f is the best objective value.
    """
    min_gens = 10
    cycles_number = 4
    iter_for_particles = int(iter_tot / cycles_number / min_gens)

    n_particles = max(5, min(iter_for_particles, x_dim * 10))

    best_x, best_f, _, _ = pso_red_v2(n_particles, iter_tot, bounds, f)
    return best_x, best_f
