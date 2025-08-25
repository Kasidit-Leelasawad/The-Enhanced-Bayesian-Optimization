import copy
import numpy as np

import copy
import numpy as np

# --- Algorithm list ---
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

# --- Configuration ---
functions     = ['Rosenbrock_f', 'Levy_f', 'Ackley_f', 'Rastrigin_f', '1_norm']
dimensions    = [2, 5, 10]
eval_limits   = [60, 150, 300]
start_indices = [12, 30, 60]
repetitions   = 5
int_ratios    = [0.2]  # define integer ratio
step          = 1 # define integer step size

test_results_latent = {}

int_var_dict = {}
for ratio in int_ratios:
    for dim in dimensions:
        int_amount = int(dim * ratio)
        int_var_dict[(ratio, dim)] = sorted(np.random.choice(dim, int_amount, replace=False).tolist())

# ✅ Main experiment loop
for ratio in int_ratios:
    ratio_key = f"r{int(ratio * 100)}"

    for func_name in functions:
        for dim, max_evals, start_idx in zip(dimensions, eval_limits, start_indices):
            dim_key = f"D{dim}"
            bounds = np.array([[-7, 7]] * dim)
            int_var = int_var_dict[(ratio, dim)]

            print(f"\n🚀 Function: {func_name} | Dimension: {dim} | int_ratio: {ratio:.1f}")

            test_results_latent.setdefault(dim_key, {})
            test_results_latent[dim_key].setdefault(ratio_key, {})
            test_results_latent[dim_key][ratio_key][func_name] = {
                'all means': {},
                'all 90': {},
                'all 10': {}
            }

            all_results = []
            shifts = np.random.uniform(-6, 6, size=(repetitions, dim))

            for alg in algorithms:
                alg_name = alg.__name__
                print(f"== {alg_name} ==")
                print(f"   Integer positions: {int_var}")

                run_results = []
                for rep in range(repetitions):
                    shift = shifts[rep].reshape((dim, 1))
                    latent_dim = dim * 3
                    linear_transform = np.random.uniform(-1, 1, size=(latent_dim, dim))

                    t_func = LatentFunction(
                        latent_func_type=func_name,
                        n_x=dim,
                        track_x=False,
                        x_shift=shift,
                        int_ratio=ratio,
                        step=step,
                        bounds=bounds,
                        int_var=int_var,
                        linear_transform=linear_transform
                    )

                    if alg_name in ['opt_DYCORS', 'opt_SRBF', 'opt_SOP']:
                        alg(t_func, dim, bounds, max_evals)
                    elif alg_name == 'opt_BO':
                        best_x, best_y = alg(t_func.eval, dim, bounds, max_evals, int_var=int_var)
                    else:
                        alg(t_func.eval, dim, bounds, max_evals)

                    t_func.best_f_list()
                    t_func.pad_or_truncate(max_evals)
                    run_results.append(copy.deepcopy(t_func.best_f_c))

                run_array = np.array(run_results)

                test_results_latent[dim_key][ratio_key][func_name][alg_name]              = run_array
                test_results_latent[dim_key][ratio_key][func_name]['all means'][alg_name] = np.mean(run_array, axis=0)
                test_results_latent[dim_key][ratio_key][func_name]['all 90'][alg_name]    = np.quantile(run_array, 0.9, axis=0)
                test_results_latent[dim_key][ratio_key][func_name]['all 10'][alg_name]    = np.quantile(run_array, 0.1, axis=0)
                all_results.append(run_array)

            combined = np.concatenate(all_results, axis=0)
            test_results_latent[dim_key][ratio_key][func_name]['mean']   = np.mean(combined, axis=0)
            test_results_latent[dim_key][ratio_key][func_name]['median'] = np.median(combined, axis=0)
            test_results_latent[dim_key][ratio_key][func_name]['q 0']    = np.max(combined, axis=0)
            test_results_latent[dim_key][ratio_key][func_name]['q 100']  = np.min(combined, axis=0)

print("\n✅ Finished running all LatentFunction experiments.")
