import numpy as np
import os
import time as time_mod


class Filter:
    def __init__(self):
        pass

    def update(self, u, z, model):
        raise NotImplementedError

    def get_params(self):
        """
        :return: dictionary of filter parameters
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def plot(self, logs, description):
        raise NotImplementedError


class Controller:
    def __init__(self):
        pass

    def get_control(self, x, t):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class Simulator:
    def __init__(self, model, filtr, ctrl):
        # Sub-classes
        self.model = model
        self.filter = filtr
        self.controller = ctrl

        # Sync Information
        self.dt = model.dt

        # Simulation properties
        self.sim_time = 20
        self.sim_info = 1

        # Flags
        self.save_history = True

        # Logs
        self.logs = None
        self.save_file = 'sim_data'

        # Checks
        self.class_checks()

    def step(self, x, t):
        # Simulate dynamics
        u = self.controller.get_control(x, t)
        x_n = self.model.prop_dynamics(x, u)
        z = self.model.get_measurement(x_n)
        self.filter.update(u, z, self.model)
        return x_n, u

    def run_sim(self):
        tic = time_mod.time()
        self.reset()
        x = self.model.initial_state()
        u = self.controller.get_control(x, 0)

        time = np.arange(0, self.sim_time + self.dt, self.dt)
        self.init_logs(len(time))

        for i, t in enumerate(time):
            self.log_data(x, u, i)
            x, u = self.step(x, t)

            if (t/self.sim_info) % 1 == 0:
                print("Sim Time: {}/{} sec".format(t, self.sim_time))

                try:
                    self.filter.plot_particles(self.logs, i)
                except AttributeError:
                    pass
        t_elapse = time_mod.time() - tic
        print("Simulation Finished in {} seconds".format(t_elapse))

        if self.save_history:
            self.logs['time'] = time
            np.savez(self.save_file, logs=self.logs, description=self.model.descriptions)
            print("Data saved to %s.npz" % self.save_file)

    def log_data(self, x, u, idx):
        if self.save_history:
            self.logs['state'][idx, :] = x.flatten()
            self.logs['control'][idx, :] = u.flatten()
            for param, val in self.filter.get_params().items():
                self.logs[param][idx, ...] = val

    def init_logs(self, N):
        if self.save_history:
            self.logs = {}
            self.logs['state'] = np.zeros([N, self.model.n])
            self.logs['control'] = np.zeros([N, self.model.k])
            for param, val in self.filter.get_params().items():
                if type(val) is np.ndarray:
                    self.logs[param] = np.zeros((N,) + val.shape)
                if type(val) is list:
                    self.logs[param] = np.zeros((N,) + len(val))
                if type(val) in (str, int, float):
                    self.logs[param] = np.zeros(N)

    def gen_plots(self):
        if self.logs is None and os.path.exists(self.save_file + '.npz'):
            data = np.load(self.save_file + '.npz')
            self.logs = data['logs'].item()
        self.filter.plot(self.logs, self.model.descriptions)

    def class_checks(self):
        n = self.model.n
        m = self.model.m
        k = self.model.k
        x0 = self.model.initial_state()
        assert x0.shape == (n, 1)
        if m == 1:
            assert type(self.model.get_measurement(x0)) is np.float64
        else:
            assert self.model.get_measurement(x0).shape == (m, 1), \
                "Mismach measurement: {} vs {}".format(self.model.get_measurement(x0).shape, (m,1))
        if k == 1:
            assert type(self.controller.get_control(x0, 0)) == np.float64
        else:
            assert self.controller.get_control(x0, 0).shape == (k, 1), \
                "Mismatch control size: {} vs {}".format(self.controller.get_control(x0, 0).shape, (k, 1))

        assert x0.dtype == np.float, 'x0 data type is {}'.format(x0)
        assert len(self.model.descriptions['states']) > 0, "Fill in model descriptions"

    def reset(self):
        self.model.reset()
        self.filter.reset()
        self.controller.reset()



