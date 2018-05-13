import numpy as np
from models import Model, DiffDrive
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


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

        # Model responsible for the dynamics and measurements
        self.model = model

    def get_controls(self, t):
        return np.array([[1, 0.2], [0.5, -0.2], [np.sin(t), 0.5]]).T

    def propagate_dynamics(self, t):
        u = self.get_controls(t)
        self.robots = self.model.prop_dynamics(self.robots, u)

    def get_measurements(self, t):
        return self.model.get_measurement(self.robots)

    def init_plot(self):
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=self.bounds['x'], ylim=self.bounds['y'])
        self.ax.set_aspect('equal')
        self.fig.canvas.draw()
        x, y, _ = self.robots
        self.state_plot = plt.plot(x, y, 'ko')[0]

    def update_plot(self):
        x,y,_ = self.robots
        self.state_plot.set_data(x, y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-10)


dt = 1e-3
model = DiffDrive(dt)
a = Arena(model)
a.init_plot()

for t in np.arange(0, 30, dt):
    a.propagate_dynamics(t)
    if 10*t % 1 == 0:
        a.update_plot()



