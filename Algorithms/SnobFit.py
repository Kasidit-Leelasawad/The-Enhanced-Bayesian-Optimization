####################
# SnobFit algorithm #
####################

def opt_SnobFit(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    n_rs = int(min(100,max(iter_tot*0.20,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = random_search_legacy(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs

    # Convert bounds to list of tuples for sk_minimize
    bounds_list = [(bounds[i, 0], bounds[i, 1]) for i in range(x_dim)]

    result, history = \
    sk_minimize(f, x_best, bounds_list, iter_, method='SnobFit')

    return result.optpar, result.optval
