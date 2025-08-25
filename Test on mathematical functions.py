''' This function class contains several mathematical functions for testing. 
    It automatically tracks function evaluations and handles early termination by 
    padding with the best value, or exceeding the budget by cutting off extra evaluations.'''

###############################################################################
#                           TestFunction Class                                #
###############################################################################
import numpy as np
import pandas as pd

class TestFunction:
    """
    A class for evaluating various test functions for optimization algorithms.
    Supports Rosenbrock, Levy, Rastrigin, Ackley, and 1-norm.
    """

    def __init__(self, func_type, n_x, int_ratio, step=0.2, spacing=None, track_x=False, x_shift=None, bounds=None, int_var=None):
        self.f_list = []
        self.x_list = []
        self.best_f = []

        self.func_type = func_type
        self.n_x = n_x
        self.spacing = step
        self.track_x = track_x
        self.int_ratio = int_ratio
        self.int_var = int_var

        # self.spacing = spacing

        if x_shift is not None:
            self.x_shift = x_shift
        else:
            self.x_shift = np.zeros((n_x, 1))

        if bounds is not None:
            self.lb = bounds[:, 0]
            self.ub = bounds[:, 1]
        else:
            self.lb = np.zeros(n_x)
            self.ub = np.ones(n_x)

        self.dim = n_x
        # int_amount = int(n_x * int_ratio)
        # self.int_var = sorted(np.random.choice(n_x, int_amount, replace=False).tolist())
        # self.cont_var = [i for i in range(n_x) if i not in self.int_var]


    def eval(self, x):
        """
        Evaluate the chosen test function. Valid types are:
        - 'Rosenbrock_f'
        - 'Levy_f'
        - 'Rastrigin_f'
        - 'Ackley_f'
        - '1_norm'
        """
        # Ensure x is a NumPy array of floats before reshaping
        x_arr = np.array(x, dtype=np.float64).reshape((-1, 1)) + self.x_shift

        if self.func_type == 'Rosenbrock_f':
            val = ((1.0 - x_arr)**2).sum() + 100.0 * ((x_arr[1:] - x_arr[:-1]**2)**2).sum()

        elif self.func_type == 'Levy_f':
            w = 1.0 + (x_arr - 1.0) / 4.0
            term1 = np.sin(np.pi * w[0])**2
            term2 = ((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0)**2)).sum()
            term3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[-1])**2)
            val = float((term1 + term2 + term3).item())

        elif self.func_type == 'Rastrigin_f':
            val = 10.0 * self.n_x + (x_arr**2 - 10.0 * np.cos(2.0 * np.pi * x_arr)).sum()
            val = float(val)

        elif self.func_type == 'Ackley_f':
            a, b, c = 20.0, 0.2, 2.0 * np.pi
            norm_sq = (x_arr**2).sum()
            term1 = -a * np.exp(-b * np.sqrt(norm_sq / self.n_x))
            term2 = -np.exp(np.cos(c * x_arr).sum() / self.n_x)
            val = float(term1 + term2 + a + np.e)

        elif self.func_type == '1_norm':
            val = float(np.abs(x_arr).sum())
        else:
            raise ValueError(f"Unsupported function type: {self.func_type}")

        self.f_list.append(val)
        if self.track_x:
            self.x_list.append(x_arr.copy())

        return val

    def best_f_list(self):
        """
        Compute best function values up to each evaluation.
        """
        accum_min = []
        current_best = float('inf')
        for val in self.f_list:
            if val < current_best:
                current_best = val
            accum_min.append(current_best)
        self.best_f = accum_min

    def pad_or_truncate(self, n_p):
        """
        Truncate or pad best_f and f_list to length n_p.
        """
        if not self.best_f:
            self.best_f_list()

        best_f_subset = self.best_f[:n_p]
        f_list_subset = self.f_list[:n_p]

        if best_f_subset:
            b_last = best_f_subset[-1]
            self.best_f_c = best_f_subset + [b_last] * (n_p - len(best_f_subset))
        else:
            self.best_f_c = [float('inf')] * n_p

        if f_list_subset:
            l_last = f_list_subset[-1]
            self.f_list_c = f_list_subset + [l_last] * (n_p - len(f_list_subset))
        else:
            self.f_list_c = [float('inf')] * n_p
