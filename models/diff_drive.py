from models import Model
import numpy as np


class DiffDrive(Model):

    def __init__(self, dt=1e-3):
        Model.__init__(self)
        self.n = 3
        self.m = 2
        self.k = 2
        self.dt = dt  # Sample time

        # Noise
        self.Q = 0.00001 * np.diag([1, 1, 1]) * self.dt
        self.R = 0.1

        # Initial State
        self.x0 = np.array([[0., 0., 0.]]).T

        # Descriptions
        self.descriptions = {'states': ['$p_x$', '$p_y$', '$\theta$'], 'controls': ['V', '\phi']}

    def prop_dynamics(self, x, u, noise=True):
        x_n = x.copy()
        v, om = u[0, :], u[1, :]
        x, y, th = x
        x_n[0] += self.dt*v*np.cos(th)
        x_n[1] += self.dt*v*np.sin(th)
        x_n[2] += self.dt*om

        if noise:
            x_n += self.process_noise(x_n.shape[1])
        return x_n

    def get_measurement(self, x, noise=True):
        y = np.zeros((self.m, x.shape[1]))
        y[0, :] = np.linalg.norm(x[0:2, :], axis=0)
        y[1, :] = x[2, :]

        if noise:
            y += self.meas_noise()
        if self.m == 1 and x.shape[1] <= 1:
            y = y[0, 0]
        return y

    def A(self, x, u):
        v, om = u.flatten()
        x, y, th = x.flatten()
        return np.array([[1, 0, -self.dt*v*np.sin(th)],
                         [0, 1,  self.dt*v*np.cos(th)],
                         [0, 0, 1]])

    def C(self, x):
        mag = np.linalg.norm(x[0:2])
        x, y, th = x.flatten()
        return np.array([[x/mag, y/mag, 0],
                         [0, 0, 1]])

    def reset(self):
        pass