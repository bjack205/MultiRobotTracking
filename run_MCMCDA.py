from filters import MCMCDA, MHKF
from models import DiffDrive, SimpleModel
from sims.sim_classes import Simulator, error_ellipse
from Arena import Arena
import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit

def plots(logs):
    # Extract from dictionary
    time = logs['time']
    state = logs['state']
    mu = logs['mu']
    sigma = logs['sigma']
    N = len(state)
    Ng = mu.shape[-1]
    n = state.shape[1]
    K = state.shape[2]

    print(state.shape)
    print(mu.shape)

    # Trajectory Plots
    plt.figure(figsize=(6, 6))
    handles = []
    for k in range(K):
        x = state[:, 0, k]
        y = state[:, 1, k]
        real = plt.plot(x, y)[0]  # State Plot
        mu_i = mu[:, :, k]
        sigma_i = sigma[:, :2, :2, k]
        est = plt.plot(mu_i[:, 0], mu_i[:, 1], 'k:')[0]   # Estimate Plot
        for j in range(10, N, 100):
            ellipse = error_ellipse(mu_i[j, :2], sigma_i[j, ...])
            plt.plot(ellipse[0, :], ellipse[1, :], 'r', linewidth=0.75)
        handles.append(real)
    plt.grid()
    plt.ylabel('x (m)')
    plt.xlabel('x (m)')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.tight_layout()
    handles.append(est)

    lgnd = ['robot {}'.format(i+1) for i in range(K)]
    lgnd.append('Estimate')
    plt.legend(handles, lgnd)

    # State Plot
    state_labels = ['x (m)', 'y (m)', 'theta (rad)']
    plt.figure()
    for j in range(n):
        plt.subplot(3, 1, j+1)
        for k in range(K):
            plt.plot(time, state[:, j, k], color='C' + str(k))
            plt.plot(time, mu[:, j, k], '--', color='C' + str(k))
        plt.grid()
        plt.xlabel('time (sec)')
        plt.ylabel(state_labels[j])

if __name__ == '__main__':
    # Set Seed
    avg_errs = []
    NUM_TRIALS = 10
    FILTER_TYPE = 'MCMCDA'
    for _ in range(NUM_TRIALS):
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
            mu0 = robots
        else:
            model = DiffDrive(dt=dt)
            model.meas_model['range'] = 1
            model.meas_model['bearing'] = 0
            model.meas_model['position'] = 0
            model.R_diag = 0.01
            model.reset()
            arena = Arena(model)
            # arena.del_robot(0)
            # arena.del_robot(0)
            arena.add_robot([-4, -2, 0], lambda t: [t, t])
            #arena.add_robot([3, -2, 0], lambda t: [np.sin(t) + 0.4, np.cos(t)])
            # arena.add_robot([-3, 4, 0], lambda t: [0.5, -.3])
            # arena.add_robot([5, -6, 0], lambda t: [0.5, -.3])
            # arena.add_robot([5, 6, 0], lambda t: [0.5, -.3])
            # arena.add_robot([-6, 6, 0], lambda t: [np.sin(t) + 0.4, np.cos(t)])
            # arena.add_robot([0, -6, 0], lambda t: [np.sin(t) - 0.4, np.cos(t)])
            arena.initial_state = arena.robots  # Keeps added robots in the area after reset
            arena.drop_rate = np.ones(arena.num_robots())*0
            if FILTER_TYPE == 'MHKF':
                mu0 = arena.robots.T.reshape(-1, 1)
            else:
                mu0 = arena.robots
        arena.reset()

        # Initial Filter
        if FILTER_TYPE == 'MCMCDA':

            filter = MCMCDA(model, mu0)
        elif FILTER_TYPE == 'MHKF':
            filter = MHKF(model, mu0)
        filter.delta = 0.1
        filter.R = np.diag([1, 1])*0.1
        filter.reset()

        # Test Filter
        z = arena.get_measurements(0)
        u = arena.get_controls(0)
        # print(timeit.timeit(lambda: filter.update(u, z, model), number=1))

        # sys.exit(0)

        # Initialize Simulator
        gif = 'MCMC_' + str(arena.num_robots())
        # gif = None
        sim = Simulator(arena, filter)
        sim.sim_time = 10
        sim.sim_info = 0.1
        sim.run_sim(gif=None) # gif)
        # sim.gen_plots()
        num_timesteps = np.arange(0, sim.sim_time + sim.dt, sim.dt).shape[0]
        avg_err = np.sum(sim.arena.errors)/ num_timesteps
        print( 'Error per timestep: ', avg_err )
        avg_errs += [avg_err]

        # Plotting
        #logs = sim.get_logs()
        #plots(logs)
        #plt.show()

    print(avg_errs)
    print('Average over ', NUM_TRIALS, "trials = ", np.mean(np.array(avg_errs)) )
