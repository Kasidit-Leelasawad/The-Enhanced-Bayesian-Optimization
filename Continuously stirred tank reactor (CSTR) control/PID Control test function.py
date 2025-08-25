import numpy as np
import copy
from scipy.integrate import odeint

# PID controller helper function
def compute_pid_control(Ks, x, x_sp, e_history):
    return PID(Ks, x, x_sp, np.array(e_history))

# Objective calculation helper function
def compute_objectives(Ca, T, Tc):
    error = np.abs(Ca) / 0.2 + np.abs(T) / 15
    u_mag = np.abs(Tc - 295) / 100  # normalized magnitude penalty
    u_cha = np.abs(np.diff(Tc)) / 100  # normalized change penalty
    return error, u_mag, u_cha

# Main simulation and control function
def J_ControlCSTR(Ks, data_res, collect_training_data=True):
    # Deep copy to avoid mutation
    Ca = copy.deepcopy(data_res['Ca_dat'])
    T = copy.deepcopy(data_res['T_dat'])
    Tc = copy.deepcopy(data_res['Tc_dat'])
    t = data_res['t']
    x     = copy.deepcopy(data_res['x0'])
    noise = data_res['noise']

    # Setpoints and bounds
    Ca_des, T_des = data_res['Ca_des'], data_res['T_des']
    Tc_lb, Tc_ub  = data_res['Tc_lb'], data_res['Tc_ub']

    e_history = []

    for i in range(len(t) - 1):
        ts = [t[i], t[i + 1]]
        x_sp = np.array([Ca_des[i], T_des[i]])

        # Compute control
        Tc[i] = compute_pid_control(Ks, x, x_sp, [[0, 0]] if i == 0 else e_history)

        # Integrate the system
        y = odeint(cstr, x, ts, args=(Tc[i],))

        # Add stochastic disturbances
        disturbance = noise * np.random.uniform(-1, 1, size=2)
        Ca[i + 1] = y[-1, 0] + disturbance[0] * 0.1
        T[i + 1] = y[-1, 1] + disturbance[1] * 5

        # Update states
        x = [Ca[i + 1], T[i + 1]]

        # Compute tracking errors
        e_history.append(x_sp - x)

    # Calculate objective penalties
    error, u_mag, u_cha = compute_objectives(np.array(e_history)[:, 0], np.array(e_history)[:, 1], Tc)

    if collect_training_data:
        data_res['Ca_train'].append(Ca)
        data_res['T_train'].append(T)
        data_res['Tc_train'].append(Tc)
        data_res['err_train'].append(error)
        data_res['u_mag_train'].append(u_mag)
        data_res['u_cha_train'].append(u_cha)
        data_res['Ks'].append(Ks)

    total_cost = np.sum(error) + np.sum(u_mag) + np.sum(u_cha)

    return total_cost

class PIDTestFunction:
    """
    A class for evaluating PID controller performance
    in a CSTR (Continuous Stirred Tank Reactor) system.
    """

    def __init__(
        self,
        data_res,
        track_x=False,
        bounds=None
    ):
        # Initialize lists to track evaluations
        self.f_list = []
        self.x_list = []
        self.best_f = []

        # Store the data resolution dictionary
        self.data_res = copy.deepcopy(data_res)

        # Reset training data collections
        self.data_res['Ca_train'] = []
        self.data_res['T_train'] = []
        self.data_res['Tc_train'] = []
        self.data_res['err_train'] = []
        self.data_res['u_mag_train'] = []
        self.data_res['u_cha_train'] = []
        self.data_res['Ks'] = []

        # Track x option
        self.track_x = track_x

        # Set bounds if provided, otherwise use default
        if bounds is not None:
            self.lb = bounds[:, 0]
            self.ub = bounds[:, 1]
        else:
            # Default bounds if not specified
            self.lb = np.zeros(3)  # Assuming 3 PID gains
            self.ub = np.ones(3)   # Normalized bounds

        # Dimensionality
        self.n_x = 3  # Typically P, I, D gains
        self.dim = self.n_x
        self.int_var = np.array([], dtype=int)
        self.cont_var = np.arange(0, self.n_x)

    def eval(self, x):
        """
        Evaluate the PID controller performance

        Parameters:
        x : array-like
            PID controller gains [Kp, Ki, Kd]

        Returns:
        float
            Total cost of the controller performance
        """
        # Create a deep copy of the data resolution to avoid mutation
        data_res_copy = copy.deepcopy(self.data_res)

        # Compute the total cost using the J_ControlCSTR function
        val = J_ControlCSTR(x, data_res_copy)

        # Store the function value
        self.f_list.append(val)

        # Track x if required
        if self.track_x:
            self.x_list.append(np.array(x).reshape((-1, 1)))

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
