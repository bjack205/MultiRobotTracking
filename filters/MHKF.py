from filters.base_filter import Filter
import numpy as np
from math import factorial, sin, cos
from scipy.linalg import block_diag
import itertools
from scipy.stats import multivariate_normal


class MHKF(Filter):

    def __init__(self, model):
        super().__init__()
        self.nr = 3  # number of robots
        self.n = model.n*self.nr  # number of states
        self.m = model.m*self.nr  # number of measurements
        self.Ng = 3  # Number of Gaussians to keep
        self.L = factorial(self.nr)
        self.M = 1

        self.mu0 = np.zeros((self.n, self.nr))
        self.mu0[:, 0] = [0, 0, 0, 1, 1, 0]
        self.sigma0 = np.repeat(np.eye(self.n)[:, :, np.newaxis], self.nr, axis=2)
        self.alpha0 = np.ones(self.Ng)/self.Ng

        self.mu = self.mu0.copy()
        self.sigma = self.sigma0.copy()
        self.alpha = self.alpha0.copy()

        self.beta = np.ones(self.M) / self.M
        self.gamma = np.ones(self.L) / self.L

    def update(self, u, z, model):
        # Reshape measurement
        y = z.T.flatten()

        # Prediction
        mu_t1t = np.zeros((self.n, self.M*self.Ng))
        sigma_t1t = np.zeros((self.n, self.n, self.M*self.Ng))
        alpha_t1t = np.zeros(self.M*self.Ng)

        for i in range(self.Ng):
            for j in range(self.M):
                k = i*self.M + j
                mu_i = self.mu[:, i].reshape(-1, model.n).T
                A_ij = [model.A(mu_i[:, k], u[:, k]) for k in range(self.nr)]
                A_ij = block_diag(*A_ij)
                Q = block_diag(model.Q, model.Q, model.Q)

                mu_t1t[:, k] = model.prop_dynamics(mu_i, u, noise=False).T.flatten()
                sigma_t1t[:, :, k] = np.linalg.multi_dot([A_ij, self.sigma[:, :, i], A_ij.T]) + Q
                alpha_t1t[k] = self.beta[j] * self.alpha[i]
        MN = self.M*self.Ng

        # Update
        mu_t1t1 = np.zeros((self.n, MN*self.L))
        sigma_t1t1 = np.zeros((self.n, self.n, MN*self.L))
        alpha_bar_t1t1 = np.zeros((MN*self.L))
        alpha_t1t1 = np.zeros((MN*self.L))

        n = model.n
        nr = self.nr
        inds = np.array([list(range(n * i, n * i + n)) for i in range(nr)])

        for k in range(MN):
            mu_k = mu_t1t[:, k]
            mu_k1 = mu_k.reshape(-1, model.n).T
            sigma_k = sigma_t1t[:, :, k]

            C = [model.C(mu_k1[:, i]) for i in range(nr)]
            C = block_diag(*C)
            R = [model.R for i in range(nr)]
            R = block_diag(*R)
            perm = itertools.permutations(range(self.nr))
            for j, p in enumerate(perm):
                i = k*self.L + j
                # Permute C and R matrix
                idx = inds[p, :].flatten()
                C_j = C[idx, :]

                # Kalman Filter Update
                K = sigma_k.dot(C.T).dot(np.linalg.inv(C.dot(sigma_k).dot(C.T) + R))
                mu_t1t1[:, i] = mu_k + K.dot(y - C_j.dot(mu_k))
                sigma_t1t1[:, :, i] = sigma_k - K.dot(C_j).dot(sigma_k)

                # Measurement posterior
                mu_y = C_j.dot(mu_k)
                sigma_y = C_j.dot(sigma_k).dot(C_j.T) + R
                p_y = multivariate_normal.pdf(y, mu_y, sigma_y)

                # Alpha value
                alpha_t1t1[i] = self.gamma[j]*alpha_t1t[k]*p_y

        # Prune
        # print(self.sigma[:3, :3, -1])
        sort_inds = np.argsort(alpha_t1t1)[-self.Ng:]
        self.mu = mu_t1t1[:, sort_inds]
        self.sigma = sigma_t1t1[:, :, sort_inds]
        self.alpha = alpha_t1t1[sort_inds]
        self.alpha /= np.sum(self.alpha)
        # print(self.sigma[:3, :3, -1])

    def get_params(self):
        return {'mu': self.mu, 'sigma': self.sigma, 'alpha': self.alpha}

    def reset(self):
        pass

