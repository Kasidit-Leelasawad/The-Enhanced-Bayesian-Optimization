def plot_performance(test_results, algorithms, functions, output_folder, dimensions, start_indices, ratio_key='r20'):
    performance = {}

    for dim, start_idx in zip(dimensions, start_indices):
        dim_key              = f"D{dim}"
        performance[dim_key] = {}

        for func_name in functions:  # ✅ ใช้ทุก function
            print(f"\n🔍 Plotting: {func_name} (D={dim})")
            for alg in algorithms:
                alg_name = alg.__name__
                trial = test_results[dim_key][ratio_key][func_name]['all means'][alg_name]
                low   = test_results[dim_key][ratio_key][func_name]['q 100']
                high  = test_results[dim_key][ratio_key][func_name]['q 0']

                perf = (high[start_idx:] - trial[start_idx:]) / (high[start_idx:] - low[start_idx:] + 1e-12)
                performance[dim_key].setdefault(alg_name, {})[func_name] = float(np.mean(perf))

            # ✅ Plot for this function
            plt.figure(figsize=(12, 6))
            for alg in algorithms:
                alg_name = alg.__name__
                trial = test_results[dim_key][ratio_key][func_name]['all means'][alg_name]
                upper = test_results[dim_key][ratio_key][func_name]['all 90'][alg_name]
                lower = test_results[dim_key][ratio_key][func_name]['all 10'][alg_name]

                x = np.arange(len(trial))
                plt.plot(trial, lw=2, label=alg_name)
                plt.fill_between(x, lower, upper, alpha=0.2)

            plt.title(f'{func_name} | Dim: {dim}')
            plt.xlabel('Iterations')
            plt.ylabel('Objective Value')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ✅ Show scores for all functions
        for func_name in functions:
            print(f"\n--- {dim_key}, {func_name} (starting from evaluation {start_idx}) ---")
            func_data = test_results[dim_key][ratio_key]
            all_mean_trajectories = func_data[func_name]['all means']
            if not all_mean_trajectories:
                print("No algorithm mean trajectories found.")
                continue

            algorithm_names = list(all_mean_trajectories.keys())
            mean_trajectories = list(all_mean_trajectories.values())

            best_mean_trajectory = np.copy(mean_trajectories[0])
            worst_mean_trajectory = np.copy(mean_trajectories[0])
            for traj in mean_trajectories[1:]:
                best_mean_trajectory = np.minimum(best_mean_trajectory, traj)
                worst_mean_trajectory = np.maximum(worst_mean_trajectory, traj)

            for alg_name, mean_traj in all_mean_trajectories.items():
                numerator = worst_mean_trajectory - mean_traj
                denominator = worst_mean_trajectory - best_mean_trajectory
                score = numerator / (denominator + 1e-9) if not np.all(denominator == 0) else np.zeros_like(numerator)
                print(f"Score for {alg_name}:  Mean Score (after start): {np.mean(score):.4f}")
