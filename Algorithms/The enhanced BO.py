import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
from torch.quasirandom import SobolEngine
warnings.filterwarnings('ignore')

!pip install deap

# Import NSGA-II libraries
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    PYMOO_AVAILABLE = True
    print("✅ pymoo NSGA-II successfully imported")
except ImportError:
    PYMOO_AVAILABLE = False
    print("❌ pymoo not available. Install with: pip install pymoo")

try:
    from deap import base, creator, tools, algorithms
    import random
    DEAP_AVAILABLE = True
    print("✅ DEAP successfully imported")
except ImportError:
    DEAP_AVAILABLE = False
    print("❌ DEAP not available. Install with: pip install deap")


class TrustRegionManager:
    """Enhanced Trust Region Manager with TuRBO-M style multi-region management."""

    def __init__(self, dim, n_trust_regions=3, length_init=1, length_min=0.25, length_max=1.6):
        self.dim = dim
        self.n_trust_regions = n_trust_regions
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max

        # Success and failure tolerances (More responsive)
        self.succtol = 3
        self.failtol = 8

        # Initialize trust region parameters
        self.length = np.ones(n_trust_regions) * length_init
        self.succcount = np.zeros(n_trust_regions, dtype=int)
        self.failcount = np.zeros(n_trust_regions, dtype=int)
        self.centers = None
        self.is_initialized = False

        # Track which trust region was used for each evaluation
        self._idx = np.zeros((0, 1), dtype=int)

    def initialize_centers(self, X, fX):
        """Initialize trust region centers using best points."""
        n_points = len(X)
        if n_points < self.n_trust_regions:
            best_idx = np.argmin(fX)
            centers_idx = [best_idx] * self.n_trust_regions
        else:
            # K-means++ style initialization
            centers_idx = [np.argmin(fX)]  # Start with best point

            for _ in range(1, self.n_trust_regions):
                distances = np.inf * np.ones(n_points)
                for idx in centers_idx:
                    dist = np.linalg.norm(X - X[idx], axis=1)
                    distances = np.minimum(distances, dist)

                probabilities = distances**2
                if probabilities.sum() > 0:
                    probabilities /= probabilities.sum()
                    next_center = np.random.choice(n_points, p=probabilities)
                else:
                    available_indices = [i for i in range(n_points) if i not in centers_idx]
                    next_center = np.random.choice(available_indices) if available_indices else centers_idx[0]
                centers_idx.append(next_center)

        self.centers = X[centers_idx].copy()
        self.is_initialized = True
        print(f"Trust regions initialized with {self.n_trust_regions} centers")
        return self.centers

    def update_trust_region(self, tr_idx, fX_next, fX_history, global_best=None):
        """Update trust region based on improvement."""
        if tr_idx >= self.n_trust_regions:
            print(f"TR-{tr_idx} index out of bounds")
            return False, False, False, None

        # Get minimum value in this trust region's history
        tr_history_idx = np.where(self._idx[:, 0] == tr_idx)[0]
        fX_min = fX_history[tr_history_idx].min() if len(tr_history_idx) > 0 else np.inf

        new_min = fX_next.min()

        # Enhanced success criteria
        improvement_threshold = 1e-3 * max(1.0, abs(fX_min))
        tr_improvement = new_min < fX_min - improvement_threshold
        global_improvement = global_best is None or new_min < global_best
        significant_tr_improvement = new_min < fX_min * 0.95
        success = significant_tr_improvement or global_improvement

        if success:
            self.succcount[tr_idx] += 1
            self.failcount[tr_idx] = 0
            print(f"   ✓ TR-{tr_idx} success (count={self.succcount[tr_idx]})")
        else:
            self.succcount[tr_idx] = 0
            self.failcount[tr_idx] += len(fX_next)
            print(f"   ✗ TR-{tr_idx} failure (count={self.failcount[tr_idx]})")

        # Check for adaptive adjustments
        adaptive_adjustments_made = False
        adjustment_type = None

        if self.succcount[tr_idx] == self.succtol:
            old_length = self.length[tr_idx]
            self.length[tr_idx] = min(2.0 * self.length[tr_idx], self.length_max)
            print(f" TR-{tr_idx} EXPANDED: {old_length:.4f} → {self.length[tr_idx]:.4f}")
            adaptive_adjustments_made = True
            adjustment_type = "success"
            self.succcount[tr_idx] = 0

        elif self.failcount[tr_idx] >= self.failtol:
            old_length = self.length[tr_idx]
            self.length[tr_idx] /= 2.0
            print(f" TR-{tr_idx} SHRUNK: {old_length:.4f} → {self.length[tr_idx]:.4f}")
            adaptive_adjustments_made = True
            adjustment_type = "failure"
            self.failcount[tr_idx] = 0

        restart_needed = self.length[tr_idx] < self.length_min
        return success, restart_needed, adaptive_adjustments_made, adjustment_type

    def restart_trust_region(self, tr_idx, X_history, y_history):
        """Restart a specific trust region."""
        print(f" Restarting TR-{tr_idx}")
        self.length[tr_idx] = self.length_init
        self.succcount[tr_idx] = 0
        self.failcount[tr_idx] = 0

        # Mark old points as inactive
        tr_idx_mask = self._idx[:, 0] == tr_idx
        self._idx[tr_idx_mask, 0] = -1

        # Find new center from global best
        best_idx = np.argmin(y_history)
        self.centers[tr_idx] = X_history[best_idx].copy()

    def add_evaluation_idx(self, idx_batch):
        """Add trust region indices for new evaluations."""
        self._idx = np.vstack((self._idx, idx_batch))

    def get_trust_region_bounds(self, tr_idx, lb, ub):
        """Get bounds for a specific trust region."""
        if not self.is_initialized:
            return lb, ub

        center = self.centers[tr_idx]
        length = self.length[tr_idx]

        tr_lb = np.clip(center - length * (ub - lb) / 2, lb, ub)
        tr_ub = np.clip(center + length * (ub - lb) / 2, lb, ub)

        return tr_lb, tr_ub

    def get_status_summary(self):
        """Get a summary of trust region status."""
        if not self.is_initialized:
            return "Trust regions not initialized"

        summary = f"\n Trust Region Status:\n"
        for i in range(self.n_trust_regions):
            active_points = np.sum(self._idx[:, 0] == i)
            summary += f"  TR-{i}: length={self.length[i]:.4f}, "
            summary += f"points={active_points}, "
            summary += f"succ={self.succcount[i]}, fail={self.failcount[i]}\n"
        return summary


class GPWrapper:
    """Wrapper for GP to match interface."""
    def __init__(self, gp):
        self.gp = gp

    def predict(self, X, return_std=False, return_cov=False):
        return self.gp.predict(X, return_std=return_std, return_cov=return_cov)

    def fit(self, X, y):
        self.gp.fit(X, y)
        return self

    def __getattr__(self, name):
        return getattr(self.gp, name)


class NSGA2EvolutionarySearch:
    """NSGA-II based optimizer for acquisition function optimization in BO."""

    def __init__(self, dim, bounds, int_var, cont_var, value_maps, backend='auto'):
        self.dim = dim
        self.bounds = bounds
        self.int_var = int_var
        self.cont_var = cont_var
        self.value_maps = value_maps

        # Auto-select backend
        if backend == 'auto':
            if PYMOO_AVAILABLE:
                self.backend = 'pymoo'
            elif DEAP_AVAILABLE:
                self.backend = 'deap'
            else:
                raise ImportError("Neither pymoo nor DEAP is available")
        else:
            self.backend = backend

        if self.backend == 'pymoo' and not PYMOO_AVAILABLE:
            raise ImportError("pymoo not available")
        if self.backend == 'deap' and not DEAP_AVAILABLE:
            raise ImportError("DEAP not available")

        print(f"Using {self.backend} for NSGA-II optimization")

    def optimize_acquisition(self, acq_func, n_suggestions=1, n_candidates=100, n_generations=20):
        """Use NSGA-II to optimize acquisition function."""
        if self.backend == 'pymoo':
            return self._optimize_pymoo(acq_func, n_suggestions, n_candidates, n_generations)
        else:
            return self._optimize_deap(acq_func, n_suggestions, n_candidates, n_generations)

    def _optimize_pymoo(self, acq_func, n_suggestions, n_candidates, n_generations):
        """Optimize using pymoo NSGA-II."""

        class AcquisitionProblem(Problem):
            def __init__(problem_self, outer_self, acq_func):
                problem_self.outer_self = outer_self
                problem_self.acq_func = acq_func

                # Set bounds
                xl, xu = [], []
                for i in range(problem_self.outer_self.dim):
                    if i in problem_self.outer_self.int_var:
                        xl.append(0)
                        xu.append(len(problem_self.outer_self.value_maps[i]) - 1)
                    else:
                        xl.append(problem_self.outer_self.bounds[i][0])
                        xu.append(problem_self.outer_self.bounds[i][1])

                # Test to determine number of objectives
                test_x = problem_self._create_test_array()
                test_result = problem_self.acq_func(test_x)
                n_obj = test_result.shape[1] if len(test_result.shape) > 1 else 1

                super().__init__(
                    n_var=problem_self.outer_self.dim,
                    n_obj=n_obj,
                    xl=np.array(xl),
                    xu=np.array(xu)
                )

            def _create_test_array(problem_self):
                """Create test array to determine n_objectives."""
                x = np.zeros((1, problem_self.outer_self.dim))
                for i in range(problem_self.outer_self.dim):
                    if i in problem_self.outer_self.int_var:
                        x[0, i] = len(problem_self.outer_self.value_maps[i]) // 2
                    else:
                        x[0, i] = (problem_self.outer_self.bounds[i][0] + problem_self.outer_self.bounds[i][1]) / 2
                return x

            def _evaluate(problem_self, X, out, *args, **kwargs):
                """Evaluate acquisition function for pymoo."""
                X_real = X.copy()
                for i in problem_self.outer_self.int_var:
                    X_real[:, i] = np.clip(np.round(X[:, i]), 0, len(problem_self.outer_self.value_maps[i]) - 1)

                acq_values = problem_self.acq_func(X_real)
                out["F"] = -acq_values

        problem = AcquisitionProblem(self, acq_func)

        algorithm = NSGA2(
            pop_size=n_candidates,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0/self.dim, eta=20)
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            verbose=False
        )

        if res.X is None:
            print("NSGA-II optimization failed, using random candidates")
            return self._generate_random_candidates(n_suggestions)

        # Get best solutions
        if len(res.X.shape) == 1:
            best_solutions = res.X.reshape(1, -1)
        else:
            best_solutions = res.X[:min(n_suggestions, len(res.X))]

        # Convert to our parameter format
        candidates = []
        for x in best_solutions:
            row = []
            for i in range(self.dim):
                if i in self.int_var:
                    idx = int(np.clip(np.round(x[i]), 0, len(self.value_maps[i]) - 1))
                    row.append(idx)
                else:
                    row.append(x[i])
            candidates.append(row)

        # Fill remaining slots if needed
        while len(candidates) < n_suggestions:
            candidates.extend(self._generate_random_candidates(1))

        return candidates[:n_suggestions]

    def _optimize_deap(self, acq_func, n_suggestions, n_candidates, n_generations):
        """Optimize using DEAP NSGA-II."""

        # Clean up any existing creator classes
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Setup DEAP
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * 10)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Register attribute generators
        for i in range(self.dim):
            if i in self.int_var:
                toolbox.register(f"attr_{i}", random.randint, 0, len(self.value_maps[i]) - 1)
            else:
                toolbox.register(f"attr_{i}", random.uniform, self.bounds[i][0], self.bounds[i][1])

        # Individual generator
        def create_individual():
            ind = creator.Individual()
            for i in range(self.dim):
                ind.append(getattr(toolbox, f"attr_{i}")())
            return ind

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        def evaluate(individual):
            x = np.array(individual).reshape(1, -1)
            acq_values = acq_func(x)
            if hasattr(acq_values, 'shape') and len(acq_values.shape) > 1:
                return tuple(acq_values[0])
            else:
                return (float(acq_values),)

        toolbox.register("evaluate", evaluate)

        # Genetic operators with bounds
        low_bounds, up_bounds = [], []
        for i in range(self.dim):
            if i in self.int_var:
                low_bounds.append(0)
                up_bounds.append(len(self.value_maps[i]) - 1)
            else:
                low_bounds.append(self.bounds[i][0])
                up_bounds.append(self.bounds[i][1])

        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                        low=low_bounds, up=up_bounds, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        low=low_bounds, up=up_bounds, eta=20.0, indpb=1.0/self.dim)
        toolbox.register("select", tools.selNSGA2)

        # Run evolution
        population = toolbox.population(n=n_candidates)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution loop
        for gen in range(n_generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=1.0)
            fitnesses = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            population = toolbox.select(population + offspring, n_candidates)

        # Get best individuals
        best_individuals = tools.selBest(population, n_suggestions)

        # Convert to our format
        candidates = []
        for ind in best_individuals:
            row = []
            for i in range(self.dim):
                if i in self.int_var:
                    row.append(int(np.clip(ind[i], 0, len(self.value_maps[i]) - 1)))
                else:
                    row.append(ind[i])
            candidates.append(row)

        # Clean up
        del creator.FitnessMulti
        del creator.Individual

        return candidates

    def _generate_random_candidates(self, n_candidates):
        """Generate random candidates as fallback."""
        candidates = []
        for _ in range(n_candidates):
            x = []
            for i in range(self.dim):
                if i in self.int_var:
                    x.append(np.random.randint(0, len(self.value_maps[i])))
                else:
                    x.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1]))
            candidates.append(x)
        return candidates


class EnhancedBayesianOptimizer:
    """Enhanced Bayesian Optimizer with TuRBO-M style trust regions and NSGA-II."""

    def __init__(self, test_func, dim, int_var=None, bounds=None, step=0.2,
                 max_evals=20, scaler_type='minmax', n_trust_regions=3,
                 use_thompson_sampling=True, batch_size=4, verbose=1,
                 use_evolution=True, beta=2.0, evolution_backend='auto'):

        self.test_func = test_func
        self.dim = dim
        self.step = step
        self.max_evals = max_evals
        self.scaler_type = scaler_type
        self.batch_size = batch_size
        self.use_thompson_sampling = use_thompson_sampling
        self.verbose = verbose
        self.use_evolution = use_evolution
        self.beta = beta
        self.evolution_backend = evolution_backend

        # Initialize components
        self._init_variable_types(int_var)
        self._init_bounds_and_maps(bounds)
        self._init_scaler()
        self._init_trust_region_manager(n_trust_regions)
        self._init_surrogate()
        self._init_optimizer()

        # Initialize history
        self.history_x = []
        self.history_y = []
        self.history_x_normalized = []
        self.current_best_y = np.inf
        self.current_best_x = None

        # Enhanced batch size parameters
        self.batch_size_min = 2
        self.batch_size_max = 10
        self.batch_size_step = 1
        self.batch_size_history = [self.batch_size]

        # Trust region count parameters
        self.n_tr_min = 1
        self.n_tr_max = 4
        self.tr_adjust_step = 1

        # Weight ratio (used in acquisition function)
        self.weight_ucb = 0.5
        self.weight_ts = 0.5

        # Performance tracking
        self.no_improvement_count = 0
        self.improvement_threshold = 1e-6
        self.stagnation_limit = 15

        self._selected_tr_indices = None

    def _print_current_status(self, iteration):
        """Print current parameter values."""
        print(f"\n🔧 ITERATION {iteration} - CURRENT PARAMETERS:")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Number of TRs: {self.trust_region_mgr.n_trust_regions}")
        print(f"   No improvement count: {self.no_improvement_count}/{self.stagnation_limit}")

    def _init_variable_types(self, int_var):
        """Initialize continuous and integer variable indices."""
        if int_var is None:
            int_amount = max(1, int(self.dim * 0.5))
            np.random.seed(42)
            self.int_var = sorted(np.random.choice(self.dim, int_amount, replace=False).tolist())
            np.random.seed()
        else:
            self.int_var = sorted(int_var)
        self.cont_var = [i for i in range(self.dim) if i not in self.int_var]

    def _init_bounds_and_maps(self, bounds):
        """Initialize bounds and value mappings."""
        if bounds is None:
            self.bounds = np.array([[0, 1]] * self.dim)
        else:
            self.bounds = bounds

        # Create value maps for integer variables
        self.value_maps = {}
        for i in self.int_var:
            lb, ub = self.bounds[i]
            values = np.round(np.arange(lb, ub + self.step, self.step), 10)
            self.value_maps[i] = values.tolist()

        # Create optimization bounds
        self.opt_bounds = []
        for i in range(self.dim):
            if i in self.int_var:
                ub_idx = len(self.value_maps[i]) - 1
                self.opt_bounds.append((0, ub_idx))
            else:
                self.opt_bounds.append(tuple(self.bounds[i]))

    def _init_scaler(self):
        """Initialize input scaler."""
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Generate representative samples for fitting
        n_fit_samples = min(1000, 10**min(self.dim, 4))
        fit_points = []
        for _ in range(n_fit_samples):
            point = []
            for i in range(self.dim):
                if i in self.int_var:
                    point.append(np.random.choice(self.value_maps[i]))
                else:
                    point.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1]))
            fit_points.append(point)

        self.scaler.fit(np.array(fit_points))

    def _init_trust_region_manager(self, n_trust_regions):
        """Initialize trust region manager."""
        self.trust_region_mgr = TrustRegionManager(
            dim=self.dim,
            n_trust_regions=n_trust_regions
        )

    def _init_surrogate(self):
        """Initialize Gaussian Process surrogate model."""
        self.kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e3), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        self.surrogate = GPWrapper(gp)

    def _init_optimizer(self):
        """Initialize NSGA-II evolution optimizer."""
        if self.use_evolution:
            try:
                self.evolution_search = NSGA2EvolutionarySearch(
                    dim=self.dim,
                    bounds=self.bounds,
                    int_var=self.int_var,
                    cont_var=self.cont_var,
                    value_maps=self.value_maps,
                    backend=self.evolution_backend
                )
                print("NSGA-II Evolution optimizer initialized successfully")
            except Exception as e:
                print(f"Warning: NSGA-II Evolution initialization failed: {e}")
                self.use_evolution = False
                self.evolution_search = None

    def normalize_inputs(self, X):
        """Normalize inputs using fitted scaler."""
        return self.scaler.transform(X)

    def denormalize_inputs(self, X_normalized):
        """Denormalize inputs using fitted scaler."""
        return self.scaler.inverse_transform(X_normalized)

    def map_params(self, params_df):
        """Map integer indices to actual values."""
        real_params = params_df.copy()
        for i in self.int_var:
            col = f'x{i}'
            real_params[col] = real_params[col].apply(
                lambda idx: self.value_maps[i][int(np.clip(round(idx), 0, len(self.value_maps[i])-1))]
            )
        return real_params

    def _generate_candidates_for_trust_region(self, n_candidates, tr_idx):
        """Generate candidates for a specific trust region."""
        if not self.trust_region_mgr.is_initialized:
            return self._generate_random_candidates(n_candidates)

        # Get trust region bounds
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)
        tr_lb_norm, tr_ub_norm = self.trust_region_mgr.get_trust_region_bounds(tr_idx, lb, ub)

        if self.use_evolution and self.evolution_search is not None:
            try:
                # Create acquisition function for this trust region
                def tr_acquisition(X_array):
                    X_real = []
                    for row in X_array:
                        x_real = []
                        for i in range(self.dim):
                            if i in self.int_var:
                                idx = int(np.clip(row[i], 0, len(self.value_maps[i]) - 1))
                                x_real.append(self.value_maps[i][idx])
                            else:
                                x_real.append(row[i])
                        X_real.append(x_real)
                    X_real = np.array(X_real)
                    X_norm = self.normalize_inputs(X_real)
                    X_norm = np.clip(X_norm, tr_lb_norm, tr_ub_norm)
                    return self._multi_objective_acquisition(X_norm)

                candidates = self.evolution_search.optimize_acquisition(
                    acq_func=tr_acquisition,
                    n_suggestions=n_candidates,
                    n_candidates=min(100, n_candidates * 10),
                    n_generations=20
                )
                return candidates

            except Exception as e:
                if self.verbose >= 1:
                    print(f"    NSGA-II failed for TR-{tr_idx}: {e}")

        # Fallback: Sobol-based generation
        try:
            sobol = SobolEngine(dimension=self.dim, scramble=True)
            pert = sobol.draw(n_candidates).numpy()
        except:
            pert = np.random.rand(n_candidates, self.dim)

        # Scale to trust region
        pert = tr_lb_norm + (tr_ub_norm - tr_lb_norm) * pert

        # Create perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(n_candidates, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim, size=len(ind))] = 1

        # Create candidates
        center = self.trust_region_mgr.centers[tr_idx]
        X_cand = np.tile(center, (n_candidates, 1))
        X_cand[mask] = pert[mask]

        # Convert to parameter space
        X_cand_real = self.denormalize_inputs(X_cand)
        candidates = []
        for row in X_cand_real:
            x = []
            for i in range(self.dim):
                if i in self.int_var:
                    values = self.value_maps[i]
                    val = min(values, key=lambda v: abs(v - row[i]))
                    idx = values.index(val)
                    x.append(idx)
                else:
                    x.append(row[i])
            candidates.append(x)

        return candidates

    def _generate_random_candidates(self, n_candidates):
        """Generate random candidates within bounds."""
        candidates = []
        for _ in range(n_candidates):
            x = []
            for i in range(self.dim):
                lb, ub = self.opt_bounds[i]
                if i in self.int_var:
                    val = np.random.randint(lb, ub + 1)
                else:
                    val = np.random.uniform(lb, ub)
                x.append(val)
            candidates.append(x)
        return candidates

    def _multi_objective_acquisition(self, candidates_norm):
        """Compute multi-objective acquisition values."""
        iteration = len(self.history_y)

        # Dynamic acquisition weighting based on progress (3 phases)
        progress = iteration / self.max_evals
        if progress < 0.3:  # First 30% - More exploration
            self.weight_ucb = 0.3
            self.weight_ts = 0.7
        elif progress < 0.7:  # Middle 40% - Balanced
            self.weight_ucb = 0.5
            self.weight_ts = 0.5
        else:  # Final 30% - More exploitation
            self.weight_ucb = 0.95
            self.weight_ts = 0.05

        candidates_norm = np.atleast_2d(candidates_norm)

        # GP-UCB with fixed beta
        mean, std = self.surrogate.predict(candidates_norm, return_std=True)
        ucb_values = -(mean + self.beta * std)

        # Thompson Sampling
        if self.use_thompson_sampling:
            ts_values = self._thompson_acquisition(candidates_norm)
            return np.column_stack([ucb_values, ts_values])
        else:
            return ucb_values.reshape(-1, 1)

    def _thompson_acquisition(self, X_candidates, n_samples=5):
        """Thompson Sampling acquisition function."""
        gp = self.surrogate.gp
        try:
            mean, cov = gp.predict(X_candidates, return_cov=True)
            cov += 1e-8 * np.eye(len(X_candidates))

            samples = []
            rng = np.random.RandomState(None)
            for _ in range(n_samples):
                try:
                    sample = rng.multivariate_normal(mean, cov)
                except np.linalg.LinAlgError:
                    std = np.sqrt(np.diag(cov))
                    sample = rng.normal(mean, std)
                samples.append(sample)

            samples = np.array(samples)
            return -np.min(samples, axis=0)

        except Exception:
            mean, std = gp.predict(X_candidates, return_std=True)
            return -mean.ravel()

    def _select_candidates_multi_tr(self, all_candidates, all_acq_values, all_tr_indices, n_suggestions):
        """Select best candidates from multiple trust regions."""
        if all_acq_values.shape[1] > 1:
            combined_acq = (
                self.weight_ucb * all_acq_values[:, 0] +
                self.weight_ts * all_acq_values[:, 1]
            )
        else:
            combined_acq = all_acq_values.ravel()

        # Select top candidates
        X_next = []
        idx_next = []
        selected_indices = set()

        for _ in range(n_suggestions):
            best_idx = None
            best_val = -np.inf

            for i in range(len(combined_acq)):
                if i not in selected_indices and combined_acq[i] > best_val:
                    best_val = combined_acq[i]
                    best_idx = i

            if best_idx is not None:
                X_next.append(all_candidates[best_idx])
                idx_next.append(all_tr_indices[best_idx])
                selected_indices.add(best_idx)

        return np.array(X_next), np.array(idx_next).reshape(-1, 1)

    def suggest(self, n_suggestions=1):
        """Suggest next points using multi-trust-region approach."""
        # Initial random sampling
        if len(self.history_x_normalized) < 2:
            candidates = self._generate_space_filling_candidates(n_suggestions)
            self._selected_tr_indices = np.zeros(n_suggestions, dtype=int)
            return pd.DataFrame(candidates, columns=[f'x{i}' for i in range(self.dim)])

        # Initialize trust regions if needed
        if not self.trust_region_mgr.is_initialized and len(self.history_x_normalized) >= 3:
            X_hist = np.vstack(self.history_x_normalized)
            y_hist = np.vstack(self.history_y).ravel()
            self.trust_region_mgr.initialize_centers(X_hist, y_hist)

        # Generate candidates from multiple trust regions
        all_candidates = []
        all_acq_values = []
        all_tr_indices = []

        base_candidates = max(30, 150 // self.trust_region_mgr.n_trust_regions)

        # Only sample from active trust regions
        active_trs = []
        for tr_idx in range(self.trust_region_mgr.n_trust_regions):
            if self.trust_region_mgr.length[tr_idx] >= self.trust_region_mgr.length_min:
                active_trs.append(tr_idx)

        if not active_trs:
            active_trs = list(range(self.trust_region_mgr.n_trust_regions))

        for tr_idx in active_trs:
            n_cand_per_tr = base_candidates
            if tr_idx < len(self.trust_region_mgr.succcount):
                if self.trust_region_mgr.succcount[tr_idx] > 0:
                    n_cand_per_tr = int(base_candidates * 1.5)

            candidates = self._generate_candidates_for_trust_region(n_cand_per_tr, tr_idx)

            if candidates:
                candidates_df = pd.DataFrame(candidates, columns=[f'x{i}' for i in range(self.dim)])
                real_candidates = self.map_params(candidates_df).values
                candidates_norm = self.normalize_inputs(real_candidates)
                acq_values = self._multi_objective_acquisition(candidates_norm)

                all_candidates.extend(candidates)
                all_acq_values.extend(acq_values)
                all_tr_indices.extend([tr_idx] * len(candidates))

        if len(all_candidates) == 0:
            candidates = self._generate_random_candidates(n_suggestions)
            self._selected_tr_indices = np.zeros(n_suggestions, dtype=int)
            return pd.DataFrame(candidates, columns=[f'x{i}' for i in range(self.dim)])

        all_candidates = np.array(all_candidates)
        all_acq_values = np.array(all_acq_values)
        all_tr_indices = np.array(all_tr_indices)

        X_next, idx_next = self._select_candidates_multi_tr(
            all_candidates, all_acq_values, all_tr_indices, n_suggestions
        )

        self._selected_tr_indices = idx_next.ravel()

        return pd.DataFrame(X_next, columns=[f'x{i}' for i in range(self.dim)])

    def _generate_space_filling_candidates(self, n_candidates):
        """Generate space-filling candidates for initial exploration."""
        try:
            sobol = SobolEngine(dimension=self.dim, scramble=True)
            sobol_points = sobol.draw(n_candidates).numpy()

            candidates = []
            for point in sobol_points:
                x = []
                for i in range(self.dim):
                    if i in self.int_var:
                        n_values = len(self.value_maps[i])
                        idx = int(point[i] * n_values)
                        idx = min(idx, n_values - 1)
                        x.append(idx)
                    else:
                        lb, ub = self.opt_bounds[i]
                        val = lb + point[i] * (ub - lb)
                        x.append(val)
                candidates.append(x)
            return candidates
        except:
            return self._generate_random_candidates(n_candidates)

    def evaluate(self, params_df):
        """Evaluate the test function at given parameters."""
        real_params = self.map_params(params_df)
        print(f"\n📋 Evaluating {len(real_params)} points:")
        print(real_params.to_string(index=False, float_format="%.5f"))

        results = []
        for _, row in real_params.iterrows():
            result = self.test_func.eval(row.values.astype(np.float64))
            results.append(result)

        return np.array(results).reshape(-1, 1)

    def observe(self, x_df, y):
        """Observe evaluation results and update model."""
        # Update history
        self.history_x.append(x_df)
        self.history_y.append(y)

        # Track improvements
        current_min = np.min(y)
        old_best = self.current_best_y

        if current_min < self.current_best_y:
            improvement = self.current_best_y - current_min
            self.current_best_y = current_min
            best_idx = np.argmin(y.ravel())
            self.current_best_x = self.map_params(x_df.iloc[[best_idx]])

            if improvement > self.improvement_threshold:
                self.no_improvement_count = 0
                print(f"    New best found! Improvement: {improvement:.6f}")
            else:
                self.no_improvement_count += 1
        else:
            self.no_improvement_count += 1

        # Normalize and store
        real_params = self.map_params(x_df)
        normalized_params = self.normalize_inputs(real_params.values)
        self.history_x_normalized.append(normalized_params)

        # Retrain GP
        X_train = np.vstack(self.history_x_normalized)
        y_train = np.vstack(self.history_y).ravel()
        self.surrogate.fit(X_train, y_train)

        # Update trust regions
        if self.trust_region_mgr.is_initialized and self._selected_tr_indices is not None:
            idx_batch = self._selected_tr_indices.reshape(-1, 1)
            self.trust_region_mgr.add_evaluation_idx(idx_batch)

            unique_tr = np.unique(self._selected_tr_indices)
            for tr_idx in unique_tr:
                if tr_idx >= self.trust_region_mgr.n_trust_regions:
                    continue

                tr_mask = self._selected_tr_indices == tr_idx
                y_tr = y[tr_mask]

                # Update center if improvement found
                if len(y_tr) > 0 and y_tr.min() < self.current_best_y:
                    best_local_idx = np.argmin(y_tr)
                    global_idx = np.where(tr_mask)[0][best_local_idx]
                    new_center = normalized_params[global_idx]
                    self.trust_region_mgr.centers[tr_idx] = new_center
                    print(f"    TR-{tr_idx} center updated to new best")

                result = self.trust_region_mgr.update_trust_region(
                    tr_idx, y_tr, y_train, global_best=self.current_best_y
                )

                if len(result) == 4:
                    success, restart_needed, adaptive_adjustments_made, adjustment_type = result
                else:
                    success, restart_needed = result
                    adaptive_adjustments_made = False
                    adjustment_type = None

                if adaptive_adjustments_made:
                    self._apply_enhanced_adaptive_adjustments(tr_idx, adjustment_type)

                if restart_needed:
                    self.trust_region_mgr.restart_trust_region(tr_idx, X_train, y_train)

    def _apply_enhanced_adaptive_adjustments(self, tr_idx, adjustment_type):
        """Apply adaptive adjustments when TR length scaling occurs."""
        print(f"\n TR-{tr_idx} triggered adaptive adjustments ({adjustment_type})")

        if adjustment_type == "success":
            # Increase batch_size
            old_batch = self.batch_size
            self.batch_size = min(self.batch_size_max, self.batch_size + self.batch_size_step)
            if self.batch_size != old_batch:
                print(f"    Increased batch_size: {old_batch} → {self.batch_size}")

            # Reduce number of TRs
            old_n_tr = self.trust_region_mgr.n_trust_regions
            new_n_tr = max(self.n_tr_min, old_n_tr - self.tr_adjust_step)
            if new_n_tr != old_n_tr:
                self._adjust_trust_regions(new_n_tr)
                print(f"    Decreased number of TRs: {old_n_tr} → {new_n_tr}")

        elif adjustment_type == "failure":
            # Reduce batch_size
            old_batch = self.batch_size
            self.batch_size = max(self.batch_size_min, self.batch_size - self.batch_size_step)
            if self.batch_size != old_batch:
                print(f"    Reduced batch_size: {old_batch} → {self.batch_size}")

            # Increase number of TRs
            old_n_tr = self.trust_region_mgr.n_trust_regions
            new_n_tr = min(self.n_tr_max, old_n_tr + self.tr_adjust_step)
            if new_n_tr != old_n_tr:
                self._adjust_trust_regions(new_n_tr)
                print(f"    Increased number of TRs: {old_n_tr} → {new_n_tr}")

    def _adjust_trust_regions(self, new_n_tr):
        """Adjust number of trust regions."""
        self.trust_region_mgr.n_trust_regions = new_n_tr
        self.trust_region_mgr.length = np.ones(new_n_tr) * self.trust_region_mgr.length_init
        self.trust_region_mgr.succcount = np.zeros(new_n_tr, dtype=int)
        self.trust_region_mgr.failcount = np.zeros(new_n_tr, dtype=int)
        self.trust_region_mgr.centers = None
        self.trust_region_mgr.is_initialized = False
        self.trust_region_mgr._idx = np.zeros((0, 1), dtype=int)

    def _restart_all_trust_regions(self):
        """Restart all trust regions when optimization stagnates."""
        print(" Restarting all trust regions due to stagnation")

        for tr_idx in range(self.trust_region_mgr.n_trust_regions):
            self.trust_region_mgr.length[tr_idx] = self.trust_region_mgr.length_init
            self.trust_region_mgr.succcount[tr_idx] = 0
            self.trust_region_mgr.failcount[tr_idx] = 0

        if len(self.history_x_normalized) > 0:
            X_hist = np.vstack(self.history_x_normalized)
            y_hist = np.vstack(self.history_y).ravel()
            self.trust_region_mgr.initialize_centers(X_hist, y_hist)

        self.batch_size = min(self.batch_size_max, self.batch_size + 2)
        print(f"   📈 Increased batch_size for exploration: {self.batch_size}")

    def optimize(self):
        """Run the optimization loop."""
        print(f" Starting Enhanced Bayesian Optimization with NSGA-II Evolution")
        print(f"   - Dimensions: {self.dim} ({len(self.cont_var)} continuous, {len(self.int_var)} integer)")
        print(f"   - Max evaluations: {self.max_evals}")
        print(f"   - Initial batch size: {self.batch_size}")
        print(f"   - Initial trust regions: {self.trust_region_mgr.n_trust_regions}")
        print(f"   - NSGA-II Evolution: {'Enabled' if self.use_evolution else 'Disabled'}")

        n_eval = 0
        iteration = 0

        # Main optimization loop
        while n_eval < self.max_evals:
            iteration += 1
            self._print_current_status(iteration)

            # Check for early stopping
            if self.no_improvement_count >= self.stagnation_limit:
                self._restart_all_trust_regions()
                self.no_improvement_count = 0

            n_suggest = min(self.batch_size, self.max_evals - n_eval)

            suggestions = self.suggest(n_suggestions=n_suggest)
            y_values = self.evaluate(suggestions)
            self.observe(suggestions, y_values)

            n_eval += n_suggest

            print(f"\n Evaluation {n_eval}/{self.max_evals}")
            print(f"   Best value: {self.current_best_y:.6f}")
            print(f"   Current batch: {[f'{val:.6f}' for val in y_values.ravel()]}")

        # Final summary
        print(f"\n✅ Optimization completed!")
        print(f"   - Total evaluations: {n_eval}")
        print(f"   - Best value found: {self.current_best_y:.6f}")
        print(f"   - Final batch size: {self.batch_size}")
        print(f"   - Final TR count: {self.trust_region_mgr.n_trust_regions}")
        print(self.trust_region_mgr.get_status_summary())

        if self.current_best_x is not None:
            print(f"\n🏆 Best parameters:")
            print(self.current_best_x.to_string(index=False, float_format="%.6f"))

        return self.current_best_x, self.current_best_y
