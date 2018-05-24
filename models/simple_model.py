from models import Model
import numpy as np


class SimpleModel(Model):

    def __init__(self, dt=1e-3):
        Model.__init__(self)
        self.n = 2
        self.m = 2
        self.k = 2
        self.dt = dt  # Sample time

        # Noise
        self.Q = 0.1 * np.diag([1, 1]) * self.dt
        self.R = 1

        # Initial State
        self.x0 = np.array([[0., 0., 0.]]).T

        # Descriptions
        self.descriptions = {'states': ['$p_x$', '$p_y$', '$\theta$'], 'controls': ['V', '\phi']}

        self.reset()

    def prop_dynamics(self, x, u, noise=True):
        x_n = x.copy()
        vx, vy = u[0, :], u[1, :]
        x_n[0, :] += self.dt * vx
        x_n[1, :] += self.dt * vy

        if noise:
            x_n += self.process_noise(x_n.shape[1])
        return x_n

    def get_measurement(self, x, noise=True):
        y = x.copy()

        if noise:
            y += self.meas_noise(x.shape[1])
        return y

    def A(self, x, u):
        return np.eye(self.n)

    def C(self, x):
        return np.eye(self.m)

    def reset(self):
        self.R = np.eye(self.m)
