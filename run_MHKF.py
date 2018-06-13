from filters.MHKF import MHKF
from models import DiffDrive, SimpleModel
from sims.sim_classes import Simulator, error_ellipse
from Arena import Arena
import numpy as np
import matplotlib.pyplot as plt
import sys



def plots(logs):
    # Extract from dictionary
    time = logs['time']
    state = logs['state']
    mu = logs['mu']
    sigma = logs['sigma']
    if 'alpha' in logs:
        alpha = logs['alpha']
    N = len(state)
    Ng = mu.shape[-1]
    n = state.shape[1]
    K = state.shape[2]

    print(state.shape)
    print(mu.shape)

    # Trajectory Plots
    plt.figure(figsize=(6, 8))
    handles = []
    for g in range(Ng):
        plt.subplot(Ng, 1, Ng - g)
        for i in range(K):
            x = state[:, 0, i]
            y = state[:, 1, i]
            real = plt.plot(x, y)[0]  # State Plot
            mu_i = mu[:, n*i:n*i+2, g]
            sigma_i = sigma[:, n*i:n*i+2, n*i:n*i+2, g]
            est = plt.plot(mu_i[:, 0], mu_i[:, 1], 'k:')[0]   # Estimate Plot
            for j in range(10, N, 100):
                ellipse = error_ellipse(mu_i[j, :], sigma_i[j, ...])
                plt.plot(ellipse[0, :], ellipse[1, :], 'r', linewidth=0.75)
            if g == 0:
                handles.append(real)
        plt.grid()
        plt.ylabel('x (m)')
        plt.xlabel('x (m)')
        plt.title('Component ' + str(Ng - g))
    plt.tight_layout()
    handles.append(est)

    lgnd = ['robot {}'.format(i+1) for i in range(K)]
    lgnd.append('Estimate')
    plt.legend(handles, lgnd)

    # Weight evolution
    if 'alpha' in logs:
        plt.figure()
        plt.plot(logs['time'], alpha)
        plt.grid()
        plt.legend(['C{}'.format(Ng-i) for i in range(Ng)])
        plt.ylabel('Component weights')
        plt.xlabel('time (sec)')

    # State Plot
    state_labels = ['x (m)', 'y (m)', 'theta (rad)']
    plt.figure()
    for j in range(n):
        plt.subplot(3, 1, j+1)
        for i in range(K):
            mu_i = mu[:, n * i:n * i + 3, -1]
            plt.plot(time, state[:, j, i], color='C' + str(i))
            plt.plot(time, mu_i[:, j], '--', color='C' + str(i))
        plt.grid()
        plt.xlabel('time (sec)')
        plt.ylabel(state_labels[j])


if __name__ == '__main__':
    # Set Seed
    np.random.seed(1)

    # Initial State
    robots = np.array([[0., 0.],
                       [1., 0.],
                       [0., 1.]]).T

    control_laws = [lambda t: np.array([np.cos(0.1 * t), np.sin(0.1 * t)]),
                    lambda t: np.array([-np.cos(0.2 * t), np.sin(0.2 * t)]),
                    lambda t: np.array([np.cos(0.1 * t), np.sin(0.2 * t)])]

    drop_rate = [0.0, 0.0, 0.0]

    # Set up Arena
    dt = 1e-2
    simple = 0
    if simple:
        model = SimpleModel(dt=dt)
        arena = Arena(model)
        arena.initial_state = robots
        arena.control_laws = control_laws
        arena.drop_rate = drop_rate
        mu0 = robots.T.reshape(-1, 1)
    else:
        model = DiffDrive(dt=dt)
        model.meas_model['range'] = 1
        model.meas_model['bearing'] = 0
        model.meas_model['position'] = 0
        model.R_diag = 0.5
        model.reset()
        arena = Arena(model)
        arena.add_robot([-4, -2, 0], lambda t: [t, t])
        arena.add_robot([3, -2, 0], lambda t: [np.sin(t)+0.4, np.cos(t)])
        # arena.add_robot([-3, 4, 0], lambda t: [0.5, -.3])
        arena.initial_state = arena.robots  # Keeps added robots in the area after reset
        mu0 = arena.robots.T.reshape(-1, 1)
    arena.reset()

    # Initial Filter
    filter = MHKF(model, mu0)
    filter.reset()

    # Test Filter
    # z = arena.get_measurements(0)
    # u = arena.get_controls(0)
    # filter.update(u, z, model)

    # Initialize Simulator
    gif = 'MHKF_5'
    # gif = None
    sim = Simulator(arena, filter)
    sim.sim_time = 10
    sim.sim_info = 0.1
    # sim.run_sim(gif=gif)
    # sim.save_gif(gif)
    # sim.gen_plots()

    # Plotting
    logs = sim.get_logs()
    plots(logs)
    plt.show()

