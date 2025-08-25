# Define your optimization algorithms
algorithms = [
    pso_red_f_v2,
    opt_direct,
    opt_BO,
    opt_de,
    opt_SnobFit,
    opt_Bobyqa,
    opt_nelder_mead,
    SS_alg,
    random_search,
    opt_powell,
    BFGS_gs,
    opt_cobyla
]

# Set optimization parameters
dimensions     = [7]  # Example dimensions for PID gains
max_evals_list = [70]
start_indices  = [14]
repetitions    = 5
Tc_lb          = 295

ratio = 0.2 # define integer ratio
ratio_key = f"r{int(ratio * 100)}"

test_results = {}
for dim, max_evals in zip(dimensions, max_evals_list):
    dim_key = f"D{dim}"
    test_results[dim_key] = {}

    test_results[dim_key][ratio_key] = {}

    # Assume PID gains bounds between 0 and 10
    bounds = np.array([[0.,10./0.2]]*3 + [[0.,10./15]]*3 + [[Tc_lb-20,Tc_lb+20]])

    test_results[dim_key][ratio_key]['J_ControlCSTR'] = {
        'all means': {},
        'all 90': {},
        'all 10': {}
    }

    all_results = []

    # Random initial shifts for PID parameters
    shifts = np.random.uniform(0, 10, size=(repetitions, dim))

    for alg in algorithms:
        alg_name = alg.__name__
        print(f"== {alg_name} optimizing J_ControlCSTR ({dim} parameters) ==")

        run_results = []

        for rep in range(repetitions):
            # Prepare initial data resolution
            data_res = process_operation()

            # Create PIDTestFunction instance
            pid_test_func = PIDTestFunction(
                data_res,
                track_x=False,
                bounds=bounds
            )

            # Define objective function wrapper using PIDTestFunction
            def objective(Ks):
                return pid_test_func.eval(Ks)

            # Run optimization algorithm
            if alg_name in ['opt_DYCORS', 'opt_SRBF', 'opt_SOP']:
                alg(objective, dim, bounds, max_evals)

            else:
                alg(objective, dim, bounds, max_evals)

            # Store best result
            # Use the initial shifted parameters as a starting point
            pid_test_func.best_f_list()
            pid_test_func.pad_or_truncate(max_evals)
            run_results.append(copy.deepcopy(pid_test_func.best_f_c))

        run_array = np.array(run_results)

        test_results[dim_key][ratio_key]['J_ControlCSTR'][alg_name] = run_array
        test_results[dim_key][ratio_key]['J_ControlCSTR']['all means'][alg_name] = np.mean(run_array, axis=0)
        test_results[dim_key][ratio_key]['J_ControlCSTR']['all 90'][alg_name]    = np.quantile(run_array, 0.9, axis=0)
        test_results[dim_key][ratio_key]['J_ControlCSTR']['all 10'][alg_name]    = np.quantile(run_array, 0.1, axis=0)

        all_results.append(run_array)

    combined = np.concatenate(all_results)
    test_results[dim_key][ratio_key]['J_ControlCSTR']['mean']   = np.mean(combined, axis=0)
    test_results[dim_key][ratio_key]['J_ControlCSTR']['median'] = np.median(combined, axis=0)
    test_results[dim_key][ratio_key]['J_ControlCSTR']['q 0']    = np.max(combined, axis=0)
    test_results[dim_key][ratio_key]['J_ControlCSTR']['q 100']  = np.min(combined, axis=0)

print(f"\n=== Finished running all J_ControlCSTR optimization experiments (ratio={ratio}) ===")
