import numpy as np
from models import Model, DiffDrive
import matplotlib.pyplot as plt
from sims.sim_classes import error_ellipse
import pickle


class Arena:
    """
    Holds the robots and is responsible for adding and removing them
    Also responsible for plotting
    """

    def __init__(self, model):
        # Bounds of the area
        self.bounds = {'x': (-10, 10), 'y': (-10, 10)}

        # Initial robots states
        # TODO: make this an argument or read from file
        self.robots = np.array([[0, 0, np.radians(0)],
                                [1, 0, np.radians(0)],
                                [0, 1, np.radians(90)]]).T

        # Control laws (defines how each robot moves, as a function of time)
        # each element of the list is a function that returns 1D numpy array
        self.control_laws = [lambda t: np.array([np.cos(0.1*t), np.sin(0.1*t)]),
                             lambda t: np.array([-np.cos(0.2*t), np.sin(0.2*t)]),
                             lambda t: np.array([np.cos(0.1*t), np.sin(0.2*t)])]

        # Drop rate: probability of the robot not reporting a measurement
        self.drop_rate = [0.0, 0.0, 0.0]

        # Model responsible for the dynamics and measurements
        self.model = model

        self.initial_state = self.robots.copy()

    def num_robots(self):
        return self.robots.shape[1]

    def reset(self):
        self.robots = self.initial_state

    def get_controls(self, t):
        """
        Return control values for each robot based on their individual control laws
        :param t: time (in seconds)
        :return: k x r numpy array (k is the number of controls and r is the number of robots)
        """
        ctrl = [law(t) for law in self.control_laws]
        return np.array(ctrl).T

    def add_robot(self, x0, control_law, drop_rate=0):
        """
        Adds a robot to the arena
        :param x0: initial states (any container than can be converted into a numpy array)
        :param control_law: function handle that accepts a single value
            and returns a 1D numpy array of k control values
        :param drop_rate: float from 0 to 1 that determines the probability of dropping the measurement
        """
        if len(x0) == self.model.n:
            x0 = np.array(x0).reshape(-1, 1)
            self.robots = np.hstack([self.robots, x0])
            self.control_laws.append(control_law)
            self.drop_rate.append(drop_rate)
        else:
            raise Warning("Robot state doesn't match")

    def del_robot(self, idx):
        """
        Deletes a robot from the arena
        :param idx: index of the robot. Can be an interable container
        """
        if not isinstance(idx, list):
            idx = [idx, ]
        for i in idx:
            self.robots = np.delete(self.robots, i, 1)
            del self.control_laws[i]
            del self.drop_rate[i]

    def check_bounds(self):
        """
        Checks if the robot is in bounds and if not, deletes it from the arena
        :return Number of robots deleted
        """
        x, y, th = self.robots
        in_bounds = (self.bounds['x'][0] < x) & (x < self.bounds['x'][1]) & \
                    (self.bounds['y'][0] < y) & (y < self.bounds['y'][1])
        inds = list(np.where(~in_bounds)[0])
        self.del_robot(inds)
        return len(inds)

    def propagate_dynamics(self, t):
        """
        Propagate the robot dynamics forward one time step for all robots
        Based on the dynamics defined in self.model
        :param t: time (seconds)
        """
        u = self.get_controls(t)
        self.robots = self.model.prop_dynamics(self.robots, u)
        return self.robots

    def get_measurements(self, t):
        """
        Gets measurements from the beacon to all of the robots
        :param t: time (seconds)
        :return: m x r numpy array of measurements (
            m is the number of measurements, r is the number of robots)
        """
        z = self.model.get_measurement(self.robots)
        measured = np.random.binomial(1, p=1-np.array(self.drop_rate))
        z = z[:, np.where(measured)[0]]
        return z

    def init_plot(self):
        """
        Initialize the live plot
        """
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=self.bounds['x'], ylim=self.bounds['y'])
        self.ax.set_aspect('equal')
        self.fig.canvas.draw()
        x, y, *_ = self.robots
        self.state_plot = plt.plot(x, y, 'ko')[0]
        self.estimate_plot = plt.plot(x, y, 'r.')[0]
        self.ellipses = [plt.plot(x, y, 'r--')[0] for i in range(self.num_robots())]
        plt.grid()

    def update_plot(self, mu=None, sigma=None):
        """
        Updates the plot (quickly)
        """
        x,y,*_ = self.robots
        self.state_plot.set_data(x, y)
        if mu is not None:
            g = -1
            mu_g = mu[:, g]
            mu_g = mu_g.reshape(-1, self.model.n).T
            mu_x, mu_y, *_ = mu_g
            self.estimate_plot.set_data(mu_x, mu_y)
            if sigma is not None:
                sigma_g = sigma[:, :, g]
                for i in range(self.num_robots()):
                    ellipse = error_ellipse(mu_g[:2, i], sigma_g[2*i:2*i+2, 2*i:2*i+2])
                    self.ellipses[i].set_data(ellipse[0, :], ellipse[1, :])

        else:
            self.estimate_plot.set_data(x*0, y*0)
            self.ellipses.set_data(x*0, y*0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-10)


if __name__ == "__main__":
    np.random.seed(1)
    dt = 1e-3
    model = DiffDrive(dt)
    model.meas_model['range'] = 0
    model.meas_model['bearing'] = 1
    model.meas_model['rel_bearing'] = 0
    model.reset()
    a = Arena(model)
    a.init_plot()

    for t in np.arange(0, 30, dt):
        a.propagate_dynamics(t)
        a.check_bounds()
        if 10*t % 1 == 0:
            a.update_plot()
            z = a.get_measurements(t)
            print(np.degrees(z))



