import numpy as np


class Model:

    def __init__(self, dt=1e-3):
        self.n = 0  # Number of states
        self.m = 0  # Number of measurements
        self.k = 0  # Number of controls
        self.Q = np.eye(self.n)  # Process Noise
        self.R = np.eye(self.n)  # Measurement Noise
        self.descriptions = {'states': [], 'controls': []}  # Description of states and control (for plotting)
        self.x0 = 0  # Initial state

    def prop_dynamics(self, x, u, noise=True):
        raise NotImplementedError

    def get_measurement(self, x, noise=True):
        raise NotImplementedError

    def initial_state(self):
        return self.x0

    def reset(self):
        raise NotImplementedError

    def A(self, x, u):
        raise NotImplementedError

    def C(self, x):
        raise NotImplementedError

    def process_noise(self, m=1):
        return np.random.multivariate_normal(np.zeros([self.n]), self.Q, m).reshape((self.n, -1))

    def meas_noise(self):
        if self.m > 1:
            return np.random.multivariate_normal(np.zeros(self.m), self.R).reshape((self.m, 1))
        else:
            return np.random.normal(0, np.sqrt(self.R))