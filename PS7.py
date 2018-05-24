from filters.MHKF import MHKF
from models.simple_model import SimpleModel
from sims.sim_classes import Simulator, error_ellipse
from Arena import Arena
import numpy as np
import matplotlib.pyplot as plt


def plots(logs):
    # Extract from dictionary
    state = logs['state']
    mu = logs['mu']
    sigma = logs['sigma']
    alpha = logs['alpha']
    N = len(state)
    Ng = mu.shape[-1]

    # Trajectory Plots
    plt.figure(figsize=(6, 8))
    handles = []
    for g in range(Ng):
        plt.subplot(Ng, 1, Ng - g)
        for i in range(3):
            x = state[:, 0, i]
            y = state[:, 1, i]
            handles.append(plt.plot(x, y)[0])
            mu_i = mu[:, 2*i:2*i+2, g]
            sigma_i = sigma[:, 2*i:2*i+2, 2*i:2*i+2, g]
            est = plt.plot(mu_i[:, 0], mu_i[:, 1], 'k--')[0]
            for j in range(0, N, 200):
                ellipse = error_ellipse(mu_i[j, :], sigma_i[j, ...])
                plt.plot(ellipse[0, :], ellipse[1, :], 'r', linewidth=0.75)
        plt.grid()
        plt.ylabel('x (m)')
        plt.xlabel('x (m)')
        plt.title('Component ' + str(Ng - g))
    plt.tight_layout()
    handles.append(est)
    lgnd = ['robot {}'.format(i+1) for i in range(3)]
    lgnd.append('Estimate')
    plt.legend(handles, lgnd)

    # Weight evolution
    plt.figure()
    plt.plot(logs['time'], alpha)
    plt.grid()
    plt.legend(['C{}'.format(Ng-i) for i in range(Ng)])
    plt.ylabel('Component weights')
    plt.xlabel('time (sec)')


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
    model = SimpleModel(dt=dt)
    arena = Arena(model)
    arena.initial_state = robots
    arena.control_laws = control_laws
    arena.drop_rate = drop_rate
    arena.reset()

    # Initial Filter
    filter = MHKF(model)
    filter.reset()

    # Initialize Simulator
    sim = Simulator(arena, filter)
    sim.sim_time = 10
    sim.sim_info = 0.1
    # sim.run_sim()
    # sim.gen_plots()

    # Plotting
    logs = sim.get_logs()
    plots(logs)
    plt.show()
