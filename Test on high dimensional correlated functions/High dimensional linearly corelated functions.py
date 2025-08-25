import numpy as np

###############################################################################
#                               LatentFunction Class                          #
###############################################################################

class LatentFunction:
    """
    A class for evaluating test functions where the input to the core
    nonlinear function is a linear combination of the actual input variables.
    Supports Rosenbrock, Levy, Rastrigin, Ackley, and 1-norm as latent functions.
    """

    def __init__(
        self,
        latent_func_type,
        n_x,
        int_ratio,
        step=1,
        spacing=None,
        track_x=False,
        x_shift=None,
        bounds=None,
        linear_transform=None,
        int_var=None
    ):
        self.f_list = []
        self.x_list = []
        self.best_f = []

        self.latent_func_type = latent_func_type
        self.n_x = n_x
        self.track_x = track_x
        self.spacing = step
        self.int_ratio = int_ratio
        self.int_var = int_var

        if x_shift is not None:
            self.x_shift = x_shift.reshape((n_x, 1))
        else:
            self.x_shift = np.zeros((n_x, 1))

        if bounds is not None:
            self.lb = bounds[:, 0].reshape((n_x, 1))
            self.ub = bounds[:, 1].reshape((n_x, 1))
        else:
            self.lb = np.zeros((n_x, 1))
            self.ub = np.ones((n_x, 1))

        self.dim = n_x
        self.int_var = np.array([], dtype=int)
        self.cont_var = np.arange(0, n_x)

        if linear_transform is not None:
            self.linear_transform = linear_transform
            self.n_latent = linear_transform.shape[0]
        else:
            self.linear_transform = np.eye(n_x)
            self.n_latent = n_x

    def _evaluate_latent_function(self, z):
        n_latent = self.n_latent

        if self.latent_func_type == 'Rosenbrock_f':
            val = ((1.0 - z)**2).sum() + 100.0 * ((z[1:] - z[:-1]**2)**2).sum()

        elif self.latent_func_type == 'Levy_f':
            w = 1.0 + (z - 1.0) / 4.0
            term1 = np.sin(np.pi * w[0])**2
            term2 = ((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0)**2)).sum()
            term3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[-1])**2)
            val = float(term1 + term2 + term3)

        elif self.latent_func_type == 'Rastrigin_f':
            val = 10.0 * n_latent + (z**2 - 10.0 * np.cos(2.0 * np.pi * z)).sum()
            val = float(val)

        elif self.latent_func_type == 'Ackley_f':
            a, b, c = 20.0, 0.2, 2.0 * np.pi
            norm_sq = (z**2).sum()
            term1 = -a * np.exp(-b * np.sqrt(norm_sq / n_latent))
            term2 = -np.exp(np.cos(c * z).sum() / n_latent)
            val = float(term1 + term2 + a + np.e)

        elif self.latent_func_type == '1_norm':
            val = float(np.abs(z).sum())
        else:
            raise ValueError(f"Unsupported latent function type: {self.latent_func_type}")

        return val

    def eval(self, x):
        x_arr = np.array(x).reshape((self.n_x, 1)) + self.x_shift
        z = self.linear_transform @ x_arr

        val = self._evaluate_latent_function(z)

        self.f_list.append(val)
        if self.track_x:
            self.x_list.append(x_arr.copy())

        return val

    def best_f_list(self):
        accum_min = []
        current_best = float('inf')
        for val in self.f_list:
            if val < current_best:
                current_best = val
            accum_min.append(current_best)
        self.best_f = accum_min

    def pad_or_truncate(self, n_p):
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

if __name__ == '__main__':
    n_x = 10
    latent_dim = 5  # The dimension of the input to the latent function

    # Define missing variables
    ratio = 0.2  # int_ratio
    step = 1     # step size
    int_var = None  # or specific indices like [0, 2, 4]

    # Define a random linear transformation matrix (n_latent x n_x)
    linear_transform_matrix = np.random.rand(latent_dim, n_x)

    # Example usage with Rosenbrock as the latent function
    latent_rosenbrock = LatentFunction(
        latent_func_type='Rosenbrock_f',
        n_x=n_x,
        track_x=True,
        int_ratio=ratio,
        step=step,
        int_var=int_var,
        x_shift=np.ones(n_x),
        bounds=np.array([[-5, 5]] * n_x),
        linear_transform=linear_transform_matrix
    )

    # Evaluate the function at a sample point
    x_sample = np.random.rand(n_x) * 10 - 5
    value = latent_rosenbrock.eval(x_sample)
    print(f"Evaluated Latent Rosenbrock at {x_sample}: {value}")

    # Evaluate multiple times
    for _ in range(20):
        x_sample = np.random.rand(n_x) * 10 - 5
        latent_rosenbrock.eval(x_sample)

    # Get the best function values found
    latent_rosenbrock.best_f_list()
    print(f"Best function values over evaluations: {latent_rosenbrock.best_f}")

    # Pad or truncate the best function values list
    latent_rosenbrock.pad_or_truncate(30)
    print(f"Padded/Truncated best function values: {latent_rosenbrock.best_f_c}")
