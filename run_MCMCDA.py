from filters import MCMCDA, MHKF
from models import DiffDrive, SimpleModel
from sims.sim_classes import Simulator, error_ellipse
from Arena import Arena
import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit

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
        mu0 = robots
    else:
        model = DiffDrive(dt=dt)
        model.meas_model['range'] = 1
        model.meas_model['bearing'] = 0
        model.meas_model['position'] = 0
        model.reset()
        arena = Arena(model)
        mu0 = arena.robots
    arena.reset()

    # Initial Filter
    filter = MCMCDA(model, mu0)
    filter.reset()

    # Test Filter
    z = arena.get_measurements(0)
    u = arena.get_controls(0)
    print(timeit.timeit(lambda: filter.update(u, z, model), number=1))



    sys.exit(0)
    # Initialize Simulator
    sim = Simulator(arena, filter)
    sim.sim_time = 25
    sim.sim_info = 0.1
    sim.run_sim()
    # sim.gen_plots()

    # Plotting
    logs = sim.get_logs()
    plots(logs)
    plt.show()