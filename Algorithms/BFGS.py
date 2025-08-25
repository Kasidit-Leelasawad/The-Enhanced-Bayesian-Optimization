######################################
# Forward finite differences
######################################

def forward_finite_diff(f, x, Delta, f_old):
    n  = np.shape(x)[0]
    x  = x.reshape((n, 1))
    dX = np.zeros((n, 1))

    for j in range(n):
        x_d_f = np.copy(x)
        x_d_f[j] = x_d_f[j] + Delta
        dX[j] = (f(x_d_f) - f_old) / Delta
    return dX

#############################
# Line search function
#############################

def line_search_f(direction, x, f, lr, grad_k, f_old, armijo_wolfe=0):
    old_f       = f_old
    new_f       = old_f + 1.
    ls_i        = 0
    lr_i        = 2. * lr
    c_1         = 1e-4
    LS_max_iter = 8

    x_i = x  # If gradient is NaN

    # Armijo line search
    if armijo_wolfe == 1:
        armijo_ = old_f - c_1 * lr_i * grad_k.T @ direction
        while new_f > armijo_ and ls_i < LS_max_iter:
            lr_i /= 2.
            x_i = x - lr_i * direction
            new_f = f(x_i)
            ls_i += 1

    # Naive line search
    elif armijo_wolfe == 0:
        while new_f > old_f and ls_i < LS_max_iter:
            lr_i /= 2.
            x_i = x - lr_i * direction
            new_f = f(x_i)
            ls_i += 1

    return x_i, ls_i, new_f

#############################
# Approximating Hessian
#############################

def Hk_f(x, x_past, grad_i, grad_i_past, Hk_past, Imatrix):
    sk  = x - x_past
    yk  = grad_i - grad_i_past
    rho = 1. / (yk.T @ sk + 1e-7)

    Hinv = (Imatrix - rho * sk @ yk.T) @ Hk_past @ (Imatrix - rho * yk @ sk.T) + rho * sk @ sk.T
    return Hinv

#############################
# First step
#############################

def BFGS_step1(f, x0, n, grad_f, Imatrix, Delta, f_old):
    grad_i      = grad_f(f, x0, Delta, f_old)
    x           = x0 - 1e-8 * grad_i
    f_new       = f(x)
    x_past      = x0.reshape((n, 1))
    grad_i_past = grad_i
    grad_i      = grad_f(f, x, Delta, f_new)
    sk          = x - x_past
    yk          = grad_i - grad_i_past
    Hk_past     = ((yk.T @ sk) / (yk.T @ yk)) * Imatrix
    return Hk_past, grad_i_past, x_past, grad_i, x, f_new

#############################
# Randomized multistart
#############################

def x0_startf(bounds, n_s, N_x):
    """
    Generates `n_s` starting points in an `N_x`-dimensional space within `bounds`
    using uniform random sampling instead of Sobol sequences.
    """
    bounds_l = np.array([bounds[i, 1] - bounds[i, 0] for i in range(len(bounds))])
    sobol_l  = np.random.uniform(0, 1, size=(n_s, N_x))  # Replaced Sobol sequence
    lb_l     = np.array([bounds[i, 0] for i in range(len(bounds))])
    x0_start = lb_l + sobol_l * bounds_l
    return x0_start

###################################
# BFGS for 'global search'
###################################

def BFGS_gs(f, N_x, bounds, max_eval):
    """
    BFGS for global search with line search. Implements a multi-start
    approach with randomized initialization.
    """
    ns       = 5
    lr       = 1.
    grad_f   = forward_finite_diff
    grad_tol = 1e-7

    # Evaluate starting points
    x0_candidates = x0_startf(bounds, ns, N_x)
    f_l           = [f(x0_candidates[xii]) for xii in range(ns)]

    f_eval = ns
    best_point = ['none', 1e15]
    ns_eval = ns
    Delta = np.sqrt(np.finfo(float).eps)

    while len(f_l) >= 1 and f_eval <= max_eval:
        minindex = np.argmin(f_l)
        x0       = x0_candidates[minindex]
        f_old    = f_l[minindex]

        # Remove used candidate
        x0_candidates = x0_candidates.tolist()
        f_l.pop(minindex)
        x0_candidates.pop(minindex)
        x0_candidates = np.asarray(x0_candidates)

        # Initialize problem
        n = np.shape(x0)[0]
        x = np.copy(x0).reshape((n, 1))
        iter_i = 0
        Imatrix = np.identity(n)

        # First step: Gradient descent
        Hk_past, grad_i_past, x_past, grad_i, x, f_old = BFGS_step1(f, x, n, grad_f, Imatrix, Delta, f_old)
        f_eval += 2 * N_x + 2

        # Optimization loop
        first_iter = True
        while np.sum(np.abs(grad_i)) > grad_tol and f_eval < max_eval:
            if not first_iter:
                grad_i = grad_f(f, x, Delta, f_old)
                f_eval += N_x
            first_iter = False

            # Compute Hessian
            Hinv   = Hk_f(x, x_past, grad_i, grad_i_past, Hk_past, Imatrix)
            x_past = x
            Df_i   = Hinv @ grad_i

            # Line search
            x_i, ls_i, new_f = line_search_f(Df_i, x, f, lr, grad_i, f_old)
            f_eval += ls_i

            # Update variables
            grad_i_past = grad_i
            Hk_past     = Hinv
            x           = x_i
            iter_i     += 1
            f_old       = new_f

            if best_point[1] > new_f:
                best_point = [x, new_f]

        # Generate more random starting points if needed
        if len(f_l) <= 0:
            ns_eval += ns
            x0_candidates = x0_startf(bounds, ns_eval, N_x)
            x0_candidates = x0_candidates[(ns_eval - ns):]
            f_l = [f(x0_candidates[xii]) for xii in range(ns)]

    return x, new_f
