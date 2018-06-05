import numpy as np
from filters import Filter
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import itertools
import random
import timeit

class BiPartite:
    def __init__(self, nu, nv):
        self.A = np.zeros((nu, nv))
        self.nu = nu
        self.nv = nv

    def add_edge(self, u, v, w=1):
        self.A[u, v] = w

    def add_edges(self, e):
        for (u, v) in e:
            self.add_edge(u, v)

    def edges(self):
        inds = np.nonzero(self.A)
        weights = self.A[inds[0], inds[1]]
        return zip(*inds, weights)

    def partitions(self, plot=False):
        """
        Calculates the unique partitions of the graph, where each partition can only have one edge per node.
        :param plot: Boolean flag to display plots of the partitions
        :return: list of partitions. Each partition is an immutable set of edges, with edge represented
                 as a (u,v,w) tuple, where u and v are node indices and w is the weight.


        Edges are 3-tuples (source, sink, weight)
        Get neighbors for each of the measurements
        Get neighbors for each of the targets
        Take the intersection
        No two edges in a partition can share a vertex!
        """
        # get list of edges (non-zero entries) as list of tuples
        edges = np.array(list(self.edges()))
        if len(edges) == 0:
            return None
        u_adj = [[tuple(lst) for lst in edges[edges[:, 0] == i, :].tolist()] + [None, ] for i in range(self.nu)]
        v_adj = [[tuple(lst) for lst in edges[edges[:, 1] == i, :].tolist()] + [None, ] for i in range(self.nv)]
        u_part = list(itertools.product(*u_adj))
        v_part = list(itertools.product(*v_adj))

        # obtain immutable sets
        filter_None = lambda part: map(lambda p: frozenset(filter(lambda x: x is not None, p)), part)
        # throw away the None sets
        u_set = set(filter_None(u_part))
        v_set = set(filter_None(v_part))

        part = u_set.intersection(v_set)
        # omega is all of your partitions. we will end up choosing just one
        omega = [[(j, k) for j, k, w in om] for om in part]
        weight = [[w for j, k, w in om] for om in part]

        if plot:
            for p in part:
                if bool(p):
                    g = BiPartite(self.nu, self.nv)
                    g.add_edges(p)
                    g.plot()
                    plt.show()
        return omega, weight

    def plot(self):
        x1 = 0
        x2 = 5
        edges = np.array(list(self.edges()))
        segs = [((x1, u), (x2, v)) for u, v in edges]

        plt.clf()
        self.plot_line_segments(segs, color='k')
        plt.plot([x1, ] * len(edges), edges[:, 0], 'bo')
        plt.plot([x2, ] * len(edges), edges[:, 1], 'bo')

    @staticmethod
    def plot_line_segments(segments, **kwargs):
        plt.plot([x for tup in [(p1[0], p2[0], None) for (p1, p2) in segments] for x in tup],
                 [y for tup in [(p1[1], p2[1], None) for (p1, p2) in segments] for y in tup], **kwargs)


class MCMCDA(Filter):

    def __init__(self, model, mu0=None):
        """
        n is state dimension
        m is the measurement dimension

        mu is initialized to zero if not passed in
        delta is a meaningful hyperparameter (Eqn. 22)
        lambda_f is the poission val
        pd is the detection probability
        n_mc is the number of Monte Carlo samples
        n_bi is the burn-in length for MC mixing

        R is a covariance window for the measurements
        """
        super().__init__()
        self.K = 3        # number of targets
        self.n = model.n  # number of states for each target
        self.m = model.m  # number of measurements for each target

        if mu0 is None:
            self.mu0 = np.zeros((self.n, self.K))  # Store K Gaussians
        else:
            self.mu0 = mu0
        self.sigma0 = np.zeros((self.n, self.n, self.K))

        self.mu = self.mu0.copy()
        self.sigma = self.sigma0.copy()

        self.delta = 0.01  # Measurement Validation Threshold
        self.lambda_f = 0.1  # False alarm rate, per unit volum, per unit time
        self.pd = 0.9  # Detection probability

        # MCMC Params
        self.n_mc = 1000
        self.n_bi = 0.2*self.n_mc

        # Tuning Params
        self.R = np.diag([1, 1])
        self.beta = None


    def update(self, u, z, model):
        """
        N is number of validated measurements (all of measurements right now)
        Construct the graph using all of these

        Adjacency matrix:

        meas \ targets   target1    target2     target3     target4     
                meas 1   1          0           0  
                meas 2
                meas 3

        K is num_targets
        mu is (state_dim x K)    
        u is (control_dim x K)
        """
        N = z.shape[1]  # number of measurements

        # Construct Bi-partite Graph
        G = BiPartite(N, self.K)

        N = G.nu
        # initialize beta values to zero
        self.beta = np.zeros((N, self.K))

        # Prediction
        mu_bar = self.mu.copy()
        sigma_bar = self.sigma.copy()
        mu_bar = model.prop_dynamics(self.mu, u, noise=False)
        for k in range(self.K):
            A_k = model.A(self.mu[:, k], u[:, k])
            sigma_bar[:, :, k] = np.linalg.multi_dot([A_k, self.sigma[:, :, k], A_k.T]) + model.Q

        # Update
        mu_update = np.zeros((self.n, N, self.K))
        sigma_update = np.zeros((self.n, self.n, N, self.K))

        # loop over each target
        for k in range(self.K):
            C_k = model.C(mu_bar[:, k])
            mu_k = mu_bar[:, k]

            # Measurement Posterior
            sigma_k = sigma_bar[:, :, k]

            # mu_y is never actually used
            mu_y = C_k.dot(mu_bar[:, k])
            sigma_y = C_k.dot(sigma_k).dot(C_k.T) + self.R

            # Expected Measurement
            yhat = model.get_measurement(mu_bar[:, k:k+1], noise=False).flatten()

            for j in range(N):
                # Measurement Innovation
                y = z[:, j]
                innov = y - yhat

                # Measurement Validation
                p_y = multivariate_normal.pdf(y, yhat, sigma_y)
                if p_y >= self.delta:
                    G.add_edge(j, k, p_y)

                # Kalman Filter Update (Eqn. 23)
                K = sigma_k.dot(C_k.T).dot(np.linalg.inv(C_k.dot(sigma_k).dot(C_k.T) + model.R))
                mu_update[:, j, k] = mu_k + K.dot(innov)
                sigma_update[:, :, j, k] = sigma_k - K.dot(C_k).dot(sigma_k)

        # Calculate the posterior of the partition
        Omega, weights = G.partitions()
        if Omega is None:
            print("No measurements passed the threshold")
            return
        p_omega = self.partition_posterior(Omega, weights, N)

        # MCMC Sample Betas
        # TODO: Implement the MCMC Sampling for the betas
        self.mcmc(G, Omega, p_omega)

        # Projection Step
        # TODO: Figure out how to best project the GMM back to a single Gaussian

    def partition_posterior(self, Omega, weights, N):
        """
        Calculate posterior of the partition (Eq 26)
        :param Omega: set of partitions at the current time step
        :param N: number of measurements at the current time step
        :return: numpy array of normalized probabilities for each posterior

        multiply weights -- P^k(u | y_{1:t-1} ), where u is current measurement -- 1:1
        """
        p = np.zeros(len(Omega))
        pd = self.pd
        for i, part in enumerate(zip(Omega, weights)):
            omega, w = part
            o = len(omega)  # size of partition |w|
            # w = np.array([w for u, v, w in omega])  # weights Pv(u|y1:t-1)
            p[i] = self.lambda_f**(N-o) * pd**o * (1-pd)**(self.K-o) * np.prod(w)
        p /= np.sum(p)
        return p

    def mcmc(self, G, Omega, p_omega):
        """
        choose a random omega (partition)

        """
        N = G.nu
        omega = random.sample(Omega, 1)[0]
        for n in range(self.n_mc):
            omega = self.mcmc_single_step(Omega, omega, p_omega, N)
            if n > self.n_bi:  # Burn in period
                for j, k in omega:
                    self.beta[j, k] += 1/(self.n_mc - self.n_bi)

    def mcmc_single_step(self, Omega, omega, p_omega, N):
        """
        Randomly sample an edge "e" from all edges "E"

        Omega is a list of "omega"s
        omega is a list of tuples
        """
        z = np.random.uniform()
        E = set(itertools.product(range(N), range(self.K)))

        # this is the self-loop transition probability; 
        # can be set to 0.5 for slower convergence
        if z < 0:
            omega_ = omega.copy()
        else:
            e = random.sample(E,1)[0]
            u_set = set(filter(lambda x: x[0] == e[0], omega))
            v_set = set(filter(lambda x: x[1] == e[1], omega))

            # deletion move
            if e in omega:
                omega_ = omega - {e}

            # addition move
            elif bool(u_set) + bool(v_set) == 0:
                omega_ = omega | {e}

            # switch move
            elif bool(u_set) + bool(v_set) == 1:
                e_ = u_set | v_set
                assert len(e_)
                omega_ = (omega | {e}) - e_
            else:
                omega_ = omega.copy()

        if omega_ in Omega:
            pi = p_omega[Omega.index(omega_)]
            pi_ = p_omega[Omega.index(omega)]
            A = min(1, pi_/pi)
            p = np.random.binomial(1, A)
            if p:
                return omega_
            else:
                return omega
        else:
            return omega

    def get_params(self):
        return {'mu': self.mu, 'sigma': self.sigma}

    def reset(self):
        self.mu = self.mu0.copy()
        self.sigma = self.sigma0.copy()

if __name__=="__main__":
    g = BiPartite(30, 20)
    g.add_edge(0, 0)
    g.add_edge(0, 1)
    g.add_edge(2, 0)
    g.add_edge(1, 1)
    Omega = g.partitions()



    # print(timeit.timeit(lambda: g.partitions(), number=1))
    # g.plot()
