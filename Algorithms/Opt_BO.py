def opt_BO(
    test_func_eval,
    dim,
    bounds,
    max_evals,
    int_var=None,
    int_ratio=0.2, #define integer ratio here
    step=1,        #define integer step size here
    batch_size=3,
    return_optimizer=False,
    random_seed=None,
    scaler_type='minmax',
    n_trust_regions=3,
    use_thompson_sampling=True,
    verbose=1,
    batch_size_min=2,
    batch_size_max=10,
    n_tr_min=1,
    n_tr_max=4,
    weight_ucb_init=0.5,
    weight_ts_init=0.5,
    stagnation_limit=10,
    adjustment_cooldown=5,
    use_evolution=True,
    beta=2.0,
    evolution_backend='auto',
):
    """
    Enhanced Bayesian Optimization with TuRBO-M style trust regions and NSGA-II Evolution.

    Parameters:
    -----------
    test_func_eval : callable
        Objective function to minimize
    dim : int
        Number of dimensions
    bounds : np.ndarray
        Bounds for each dimension (shape: dim x 2)
    max_evals : int
        Maximum number of function evaluations
    int_var : list, optional
        List of indices for integer variables
    int_ratio : float
        Ratio of integer variables if int_var not specified
    step : float
        Step size for integer variables
    batch_size : int
        Initial batch size for parallel evaluations
    return_optimizer : bool
        Whether to return the optimizer object
    random_seed : int, optional
        Random seed for reproducibility
    scaler_type : str
        Type of scaler ('minmax' or 'standard')
    n_trust_regions : int
        Initial number of trust regions
    use_thompson_sampling : bool
        Whether to use Thompson Sampling
    verbose : int
        Verbosity level (0=quiet, 1=normal, 2=detailed)
    batch_size_min : int
        Minimum batch size
    batch_size_max : int
        Maximum batch size
    n_tr_min : int
        Minimum number of trust regions
    n_tr_max : int
        Maximum number of trust regions
    weight_ucb_init : float
        Initial weight for UCB acquisition
    weight_ts_init : float
        Initial weight for Thompson Sampling
    stagnation_limit : int
        Number of iterations without improvement before restart
    adjustment_cooldown : int
        Cooldown period between adjustments (not used in current implementation)
    use_evolution : bool
        Whether to use NSGA-II Evolution optimizer
    beta : float
        Fixed beta value for GP-UCB
    evolution_backend : str
        Backend for NSGA-II ('auto', 'pymoo', or 'deap')

    Returns:
    --------
    best_x : pd.DataFrame
        Best parameters found
    best_y : float
        Best objective value found
    optimizer : EnhancedBayesianOptimizer (optional)
        The optimizer object if return_optimizer=True
    """
    import numpy as np

    class WrappedTestFunc:
        def __init__(self, evaluator):
            self.eval = evaluator.eval if hasattr(evaluator, 'eval') else evaluator

    wrapped_func = WrappedTestFunc(test_func_eval)

    if random_seed is not None:
        np.random.seed(random_seed)

    if int_var is None:
        int_amount = max(1, int(dim * int_ratio))
        int_var = sorted(np.random.choice(dim, int_amount, replace=False).tolist())

    if bounds.shape != (dim, 2):
        raise ValueError(f"[opt_BO] Bounds shape {bounds.shape} does not match dim={dim}")
    if not all(0 <= idx < dim for idx in int_var):
        raise ValueError(f"[opt_BO] Integer variable indices {int_var} must be in range [0, {dim-1}]")

    print(f"🔧 Initializing Enhanced Bayesian Optimizer with NSGA-II Evolution")
    print(f"   - Dimensions: {dim}")
    print(f"   - Integer vars: {int_var}")
    print(f"   - Continuous vars: {[i for i in range(dim) if i not in int_var]}")
    print(f"   - Max evaluations: {max_evals}")
    print(f"   - Initial batch size: {batch_size} (range: {batch_size_min}-{batch_size_max})")
    print(f"   - Initial trust regions: {n_trust_regions} (range: {n_tr_min}-{n_tr_max})")
    print(f"   - Initial weight ratio: UCB={weight_ucb_init:.2f}, TS={weight_ts_init:.2f}")
    print(f"   - Early stopping: {stagnation_limit} iterations without improvement")
    print(f"   - Thompson Sampling: {use_thompson_sampling}")
    print(f"   - NSGA-II Evolution: {use_evolution} (backend: {evolution_backend})")
    print(f"   - Beta (GP-UCB): {beta:.3f}")

    # Create optimizer
    optimizer = EnhancedBayesianOptimizer(
        test_func=wrapped_func,
        dim=dim,
        int_var=int_var,
        bounds=bounds,
        step=step,
        max_evals=max_evals,
        batch_size=batch_size,
        scaler_type=scaler_type,
        n_trust_regions=n_trust_regions,
        use_thompson_sampling=use_thompson_sampling,
        verbose=verbose,
        use_evolution=use_evolution,
        beta=beta,
        evolution_backend=evolution_backend,
    )

    # Set adaptive parameters
    optimizer.batch_size_min = batch_size_min
    optimizer.batch_size_max = batch_size_max
    optimizer.n_tr_min = n_tr_min
    optimizer.n_tr_max = n_tr_max
    optimizer.weight_ucb = weight_ucb_init
    optimizer.weight_ts = weight_ts_init
    optimizer.stagnation_limit = stagnation_limit

    # Note: adjustment_cooldown is not used in the current implementation
    if hasattr(optimizer, 'adjustment_cooldown'):
        optimizer.adjustment_cooldown = adjustment_cooldown

    print(f"✅ Enhanced adaptive parameters configured:")
    print(f"   - Batch size range: {batch_size_min}-{batch_size_max}")
    print(f"   - TR count range: {n_tr_min}-{n_tr_max}")
    print(f"   - Adaptive TR thresholds: Success={optimizer.trust_region_mgr.succtol}, Failure={optimizer.trust_region_mgr.failtol}")
    print(f"   - Stagnation limit: {stagnation_limit}")
    print(f"   - Beta value: {beta:.3f}")
    print(f"   - Evolution backend: {evolution_backend}")

    # Run optimization
    best_x, best_y = optimizer.optimize()

    print(f"\n🏁 Enhanced optimization with NSGA-II Evolution complete!")
    print(f"   - Total evaluations: {sum(len(y) for y in optimizer.history_y)}")
    print(f"   - Best value: {best_y:.6f}")
    print(f"   - Final batch size: {optimizer.batch_size}")
    print(f"   - Final TR count: {optimizer.trust_region_mgr.n_trust_regions}")
    print(f"   - Final weight ratio: UCB={optimizer.weight_ucb:.2f}, TS={optimizer.weight_ts:.2f}")
    print(f"   - Beta value used: {beta:.3f}")
    print(f"   - Stagnation periods: {optimizer.no_improvement_count}")

    if return_optimizer:
        return best_x, best_y, optimizer
    else:
        return best_x, best_y
