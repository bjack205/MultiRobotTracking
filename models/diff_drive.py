from models import Model
import numpy as np


class DiffDrive(Model):

    def __init__(self, dt=1e-3):
        Model.__init__(self)
        self.n = 3
        self.m = 1
        self.k = 2
        self.dt = dt  # Sample time

        # Noise
        self.Q = 0.1 * np.diag([1, 1, 1]) * self.dt
        self.R_diag = 1
        self.R = self.R_diag

        # Initial State
        self.x0 = np.array([[0., 0., 0.]]).T

        # Descriptions
        self.descriptions = {'states': ['$p_x$', '$p_y$', '$\theta$'], 'controls': ['V', '\phi']}

        # Beacon location
        self.beacons = np.array([[-10, -10]]).T, np.array([[10, -10]]).T

        # Measurement model
        self.meas_model = {'range': 1, 'bearing': 0, 'rel_bearing': 0, 'position': 0}
        self.reset()

    def prop_dynamics(self, x, u, noise=True):
        x_n = x.copy()
        v, om = u[0, :], u[1, :]
        px, py, th = x
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
        diff, mag = self.vec_to_beacon(x)

        # Append measurements according to active model
        y = []
        if self.meas_model['range']:
            rho = np.vstack(mag)
            y.append(rho)

        if self.meas_model['bearing']:
            phi = np.arctan2(diff[1:2, :], diff[0:1, :])
            y.append(phi)
            raise NotImplementedError

        if self.meas_model['rel_bearing']:
            phi = np.arctan2(diff[1:2, :], diff[0:1, :]) - x[2:3, :]
            y.append(phi)
            raise NotImplementedError

        if self.meas_model['position']:
            p = x[0:2, :]
            y.append(p)

        y = np.vstack(y)

        assert y.shape == (self.m, x.shape[1])
        if noise:
            y += self.meas_noise()
        return y

    def vec_to_beacon(self, x):
        diff = [x[0:2, :] - self.beacons[0],
                x[0:2, :] - self.beacons[1]]
        mag = [np.linalg.norm(diff[0], axis=0, keepdims=True),
               np.linalg.norm(diff[1], axis=0, keepdims=True)]
        return diff, mag

    def A(self, x, u):
        v, om = u.flatten()
        x, y, th = x.flatten()
        return np.array([[1, 0, -self.dt*v*np.sin(th)],
                         [0, 1,  self.dt*v*np.cos(th)],
                         [0, 0, 1]])

    def C(self, x):
        x = x.reshape(-1, 1)
        diff, mag = self.vec_to_beacon(x)

        C = []
        if self.meas_model['range']:
            C_rho = np.zeros([len(diff), self.n])
            for i in range(len(diff)):
                C_rho[i, 0:2] = (diff[i]/mag[i]).flatten()
            C.append(C_rho)

        if self.meas_model['position']:
            C_pos = np.array([[1, 0, 0],
                              [0, 1, 0]])
            C.append(C_pos)

        C = np.vstack(C)
        return C

    def reset(self):
        if self.meas_model['position'] > 0:
            self.meas_model['position'] = 2
        if self.meas_model['range'] > 0:
            self.meas_model['range'] = len(self.beacons)
        m = sum([val for val in self.meas_model.values()])
        self.m = m
        if m == 1:
            self.R = self.R_diag
        else:
            self.R = self.R_diag*np.eye(self.m)
