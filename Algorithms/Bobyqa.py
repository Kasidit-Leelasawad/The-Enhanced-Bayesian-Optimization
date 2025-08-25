####################
# Bobyqa  algorithm #
####################

def opt_Bobyqa(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    n_rs = int(min(100, max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = random_search_legacy(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs

    result, history = \
    sk_minimize(f, x_best, bounds, iter_, method='Bobyqa')

    return result.optpar, result.optval
