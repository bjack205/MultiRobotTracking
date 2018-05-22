import numpy as np
from models import Model, DiffDrive
import matplotlib.pyplot as plt


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
        self.robots = np.array([[-2, -2, np.radians(0)],
                                [2, -2, np.radians(30)],
                                [3, 0, np.radians(90)]]).T

        # Control laws (defines how each robot moves, as a function of time)
        # each element of the list is a function that returns 1D numpy array
        self.control_laws = [lambda t: np.array([1, .2]),
                             lambda t: np.array([0.5, -0.2]),
                             lambda t: np.array([np.sin(t), 0.5]).T]

        # Drop rate: probability of the robot not reporting a measurement
        self.drop_rate = [0.1, 0.2, 0]

        # Model responsible for the dynamics and measurements
        self.model = model

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
        x, y, _ = self.robots
        self.state_plot = plt.plot(x, y, 'ko')[0]
        plt.grid()

    def update_plot(self):
        """
        Updates the plot (quickly)
        """
        x,y,_ = self.robots
        self.state_plot.set_data(x, y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-10)


if __name__ == "__main__":
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
        if np.isclose(t, 5):
            a.add_robot([0, 0, 0], lambda t: np.array([np.sin(t)*2, 2*np.cos(t)]))
        if np.isclose(t, 10):
            a.del_robot(1)



