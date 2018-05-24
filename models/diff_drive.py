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
        self.Q = 0.1 * np.diag([1, 1, 1]) * self.dt
        self.R = 1

        # Initial State
        self.x0 = np.array([[0., 0., 0.]]).T

        # Descriptions
        self.descriptions = {'states': ['$p_x$', '$p_y$', '$\theta$'], 'controls': ['V', '\phi']}

        # Beacon location
        self.beacon = np.array([[-10, -10]]).T

        # Measurement model
        self.meas_model = {'range': 1, 'bearing': 1, 'rel_bearing': 0}
        self.reset()

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
        # Extract states
        px, py, th = x

        # Calculate difference vector and magnitude
        diff = x[0:2, :] - self.beacon
        mag = np.linalg.norm(self.beacon - x[0:2, :], axis=0, keepdims=True)
        mag[np.isclose(mag, 0)] = 1e-6

        # Append measurements according to active model
        y = []
        if self.meas_model['range']:
            rho = mag
            y.append(rho)

        if self.meas_model['bearing']:
            phi = np.arctan2(diff[1:2, :], diff[0:1, :])
            y.append(phi)

        if self.meas_model['rel_bearing']:
            phi = np.arctan2(diff[1:2, :], diff[0:1, :]) - x[2:3, :]
            y.append(phi)

        y = np.vstack(y)

        assert y.shape == (self.m, x.shape[1])
        if noise:
            y += self.meas_noise()
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
        m = sum([val for val in self.meas_model.values()])
        self.m = m
        self.R = 0.1*np.eye(self.m)*0.001