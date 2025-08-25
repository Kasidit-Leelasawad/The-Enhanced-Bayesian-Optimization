import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.integrate import odeint

# Utility function for plotting repeated trajectories
def plot_repeated_trajectories(ax, t, data, desired, color, ylabel, label, ylim=None):
    repetitions = data.shape[0]
    alphas = [(repetitions - float(i)) / repetitions for i in range(repetitions)]
    for i in range(repetitions):
        ax.plot(t, data[i, :], '-', lw=1, color=color, alpha=alphas[i])
    ax.step(t, desired, '--', lw=1.5, color='black', label='Desired')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (min)')
    ax.legend([label, 'Desired'], loc='best')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True)

# Training plots
def training_plot(data_plot, data_res, repetitions):
    t = data_plot['t']
    Tc_train = np.array(data_plot['Tc_train'])
    Ca_train = np.array(data_plot['Ca_train'])
    T_train = np.array(data_plot['T_train'])
    Ca_des = data_res['Ca_des']
    T_des = data_res['T_des']

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    plot_repeated_trajectories(axes[0], t, Ca_train, Ca_des, 'r', 'A (mol/m³)', 'Concentration of A', ylim=[0.75, 0.95])
    axes[0].set_title('Training Plots')

    plot_repeated_trajectories(axes[1], t, T_train, T_des, 'c', 'T (K)', 'Reactor Temperature', ylim=[317, 335])

    for i in range(repetitions):
        axes[2].step(t[1:], Tc_train[i, :], 'b--', lw=1, alpha=(repetitions - float(i)) / repetitions)
    axes[2].set_ylabel('Cooling T (K)')
    axes[2].set_xlabel('Time (min)')
    axes[2].legend(['Jacket Temperature'], loc='best')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# Convergence plots
def plot_convergence(Xdata, best_Y=None, Objfunc=None):
    if best_Y is None:
        best_Y, f_best = [], 1e8
        for x in Xdata:
            f_val = Objfunc(x, collect_training_data=False)
            f_best = min(f_best, f_val)
            best_Y.append(f_best)
        best_Y = np.array(best_Y)

    distances = np.sqrt(np.sum((np.diff(Xdata, axis=0))**2, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(distances, '-ro')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('d(x[n], x[n-1])')
    axes[0].set_title('Distance between consecutive samples')
    axes[0].grid(True)

    axes[1].plot(best_Y, '-o')
    axes[1].set_title('Best sample value')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Best Y')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# Utility for uncertainty plots
def plot_with_uncertainty(ax, t, data, desired, color, ylabel, label):
    median = np.median(data, axis=1)
    ax.plot(t, median, '-', color=color, lw=2)
    ax.fill_between(t, np.min(data, axis=1), np.max(data, axis=1), color=color, alpha=0.2)
    ax.step(t, desired, '--', color='black', lw=1.5, label='Desired')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (min)')
    ax.legend([label, 'Desired'], loc='best')
    ax.grid(True)

# Control action performance plots
def plot_u_result(u_plt, data_plot, Ca_des, T_des, repetitions):
    t, noise, n = data_plot['t'], data_plot['noise'], data_plot['n']
    u_plt = np.array(u_plt).reshape(-1)

    Ca_dat, T_dat, Tc_dat = [], [], []
    for _ in range(repetitions):
        Ca, T, x = deepcopy(data_plot['Ca_dat']), deepcopy(data_plot['T_dat']), deepcopy(data_plot['x0'])
        Ca_run, T_run = [x[0]], [x[1]]
        for Tc in u_plt:
            y = odeint(cstr, x, [0, t[1]-t[0]], args=(Tc,))[-1]
            x = y + noise * np.random.uniform(-1, 1, 2) * np.array([0.1, 5])
            Ca_run.append(x[0])
            T_run.append(x[1])
        Ca_dat.append(Ca_run)
        T_dat.append(T_run)
        Tc_dat.append(u_plt)

    Ca_dat, T_dat, Tc_dat = np.array(Ca_dat).T, np.array(T_dat).T, np.array(Tc_dat).T

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    plot_with_uncertainty(axes[0], t, Ca_dat, Ca_des, 'r', 'A (mol/m³)', 'Concentration of A')
    axes[0].set_title('Control Actions Performance')

    plot_with_uncertainty(axes[1], t, T_dat, T_des, 'c', 'T (K)', 'Reactor Temperature')

    axes[2].step(t[1:], np.median(Tc_dat, axis=1), 'b--', lw=2)
    axes[2].set_ylabel('Cooling T (K)')
    axes[2].set_xlabel('Time (min)')
    axes[2].legend(['Jacket Temperature'], loc='best')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
