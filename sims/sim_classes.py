import numpy as np
import os
import time as time_mod


class Controller:
    def __init__(self):
        pass

    def get_control(self, x, t):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class Simulator:
    def __init__(self, arena, filtr):
        # Sub-classes
        self.arena = arena
        self.filter = filtr

        # Sync Information
        self.dt = arena.model.dt

        # Simulation properties
        self.sim_time = 20
        self.sim_info = 1

        # Flags
        self.save_history = True

        # Logs
        self.logs = None
        self.save_file = 'sim_data'

    def step(self, x, t):
        # Simulate dynamics
        u = self.arena.get_controls(t)
        x_n = self.arena.propagate_dynamics(t)
        z = self.arena.get_measurements(t)
        self.filter.update(u, z, self.arena.model)
        return x_n, u

    def run_sim(self):
        tic = time_mod.time()
        self.reset()
        x = self.arena.initial_state
        u = self.arena.get_controls(0)

        time = np.arange(0, self.sim_time + self.dt, self.dt)
        self.init_logs(len(time))
        self.arena.init_plot()

        for i, t in enumerate(time):
            self.log_data(x, u, i)
            x, u = self.step(x, t)

            if (t/self.sim_info) % 1 == 0:
                print("Sim Time: {}/{} sec".format(np.round(t, 2), np.round(self.sim_time, 2)))
                self.arena.update_plot(mu=self.get_mu(i), sigma=self.get_sigma(i))

        t_elapse = time_mod.time() - tic
        print("Simulation Finished in {} seconds".format(t_elapse))

        if self.save_history:
            self.logs['time'] = time
            np.savez(self.save_file, logs=self.logs)
            print("Data saved to %s.npz" % self.save_file)

    def get_mu(self, i):
        if 'mu' in self.logs:
            return self.logs['mu'][i, ...]

    def get_sigma(self, i):
        if 'sigma' in self.logs:
            return self.logs['sigma'][i, ...]

    def log_data(self, x, u, idx):
        if self.save_history:
            self.logs['state'][idx, :] = x
            self.logs['control'][idx, :] = u
            for param, val in self.filter.get_params().items():
                self.logs[param][idx, ...] = val

    def init_logs(self, N):
        if self.save_history:
            self.logs = {}
            x0 = self.arena.initial_state
            u0 = self.arena.get_controls(0)
            self.logs['state'] = np.zeros((N,) + x0.shape)
            self.logs['control'] = np.zeros((N,) + u0.shape)
            for param, val in self.filter.get_params().items():
                if type(val) is np.ndarray:
                    self.logs[param] = np.zeros((N,) + val.shape)
                if type(val) is list:
                    self.logs[param] = np.zeros((N,) + len(val))
                if type(val) in (str, int, float):
                    self.logs[param] = np.zeros(N)

    def get_logs(self):
        if self.logs is None and os.path.exists(self.save_file + '.npz'):
            data = np.load(self.save_file + '.npz')
            self.logs = data['logs'].item()
        return self.logs

    def gen_plots(self):
        if self.logs is None and os.path.exists(self.save_file + '.npz'):
            data = np.load(self.save_file + '.npz')
            self.logs = data['logs'].item()
        self.filter.plot(self.logs, self.model.descriptions)


    def reset(self):
        self.arena.reset()
        self.filter.reset()


def error_ellipse(mu, sigma, p=0.95):
    mu = mu.reshape(-1, 1)
    eta = 1/(2*np.pi*np.sqrt(np.linalg.det(sigma)))
    eps = (1-p)*eta
    alpha = -2*np.log(eps/eta)

    E = np.linalg.inv(sigma)/alpha
    R = np.linalg.cholesky(E)
    t = np.linspace(0, 2*np.pi, 100)
    z = np.vstack([np.cos(t), np.sin(t)])
    ellipse = np.linalg.solve(R, z) + mu
    return ellipse
