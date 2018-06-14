import numpy as np 
import matplotlib.pyplot as plt 
import math
import pdb
import scipy.linalg

def generate_whitened_noise(cov,dim):
	return np.squeeze( np.matmul(scipy.linalg.sqrtm(cov),np.random.randn(dim)) )


class MultiHypothesisKalmanFilter():
	def __init__(self, num_timesteps):
		self.num_timesteps = num_timesteps
		self.num_targets = 3
		self.R = 10. * np.eye(self.num_targets*2) # essentially, 3 block diagonal identity mats I_{2 x 2}
		self.Q = 0.1 * np.eye(self.num_targets*2)
		self.dt = 1.
		self.weights = np.array([0.5,0.25,0.25])
		self.true_states = [ np.array([0, 0, 10, 0, 20, 0]) ]
		self.measurements = []

		self.Ctrue = np.zeros((3,3))
		self.Ctrue[0,0] = 1
		self.Ctrue[1,2] = 1
		self.Ctrue[2,1] = 1
		self.Ctrue = self.duplicate_cols(self.Ctrue)

	def generate_sequence(self):
		""" 
		Run simulation of the dynamics and measurement models
		for some number of timesteps.
		"""
		x_t = self.true_states[0] # initial state
		y_t = self.h(x_t, add_noise=True)
		self.measurements += [y_t]

		for t in range(self.num_timesteps-1):
			x_t = self.dynamics(x_t, t, add_noise = True)
			self.true_states += [x_t]
			y_t = self.h(x_t, add_noise=True)
			self.measurements += [y_t]

		return self.true_states, self.measurements


	def h(self, x_t, add_noise=True):
		""" measurement model for generating y """
		y_t = self.Ctrue.dot( x_t )
		if add_noise:
			y_t += generate_whitened_noise(self.R,6)
		return y_t

	def get_initial_state_estimate(self):
		""" """
		pred_state = np.zeros((self.num_targets,6))
		pred_cov = np.zeros((self.num_targets,6,6))
		# loop over the hypotheses (which are true targets initially)
		for i in range(self.num_targets):
			pred_state[i] = 0.5 * np.random.rand(6) + self.true_states[0]
			pred_cov[i] = 100 * np.eye(6)

		return pred_state, pred_cov


	def duplicate_cols(self, perm_mat):
		""" 
		We want to permute the (x,y) coords together 
		So we duplicate each col
		"""
		assert perm_mat.shape[0] == perm_mat.shape[1]
		assert perm_mat.shape[0] == self.num_targets
		dup_mat = np.zeros((self.num_targets*2,self.num_targets*2))
		for i in range(self.num_targets):
			j = np.argwhere(perm_mat[i] == 1)[0,0]
			dup_mat[i*2,j*2] = 1
			dup_mat[i*2+1,j*2+1] = 1
		return dup_mat


	def generate_permutation_matrices(self):
		"""
		Generate N! permutation matrices
		(6 permutation matrices here since N=3)

		3     2     1
		3     1     2
		2     3     1
		2     1     3
		1     3     2
		1     2     3

		"""
		perm_mats = []
		for i in range(3):
			for j in range(3):
				if j == i:
					continue
				for k in range(3):
					if (k == j) or (k==i):
						continue
					perm_mat = np.zeros((3,3))
					perm_mat[0,i] = 1
					perm_mat[1,j] = 1
					perm_mat[2,k] = 1
					perm_mat = self.duplicate_cols(perm_mat)
					perm_mats += [ perm_mat ]
					#print perm_mat
					#print
		return perm_mats


	def dynamics(self, x_t, t, add_noise):
		""" 
		x_{t+1}^i = x_t^i + u_t^i + w_t^i 

		Q is our process noise
		"""
		x_tplus1 = x_t + np.hstack([self.compute_u_t(t,1),
									self.compute_u_t(t,2),
									self.compute_u_t(t,3)])
		# add white Gaussian noise
		if add_noise:
			x_tplus1 += generate_whitened_noise(self.Q, 6)
		return x_tplus1


	def compute_u_t(self, t, gauss_idx):
		if gauss_idx == 1: # u_1_t
			return np.array([np.cos(0.1*t), np.sin(0.1*t)])
		elif gauss_idx == 2: # u_2_t
			return np.array([-np.cos(0.2*t), np.sin(0.2*t)])
		elif gauss_idx == 3: # u_3_t
			return np.array([np.cos(0.1*t), np.sin(0.2*t)])
		else:
			print 'Undefined Gaussian component index. Quitting...'
			quit()


	def gauss_pdf(self, x, mu, cov):
		"""
		x is the point at which we evaluate
		mu is the mean of the Gaussian
		cov is the covariance of the Gaussian
		"""
		n = cov.shape[0]
		exponent = -0.5 * (x-mu).T.dot(scipy.linalg.inv(cov)).dot(x-mu )
		eta = ((2 * np.pi) ** (-n/2.)) * (scipy.linalg.det(cov) ** (-0.5) )
		return eta * np.exp(exponent)


	def run_mhkf_step(self, pred_mu, pred_cov, t, y_true):
		"""
		keep only the first three most likely Gaussian components at each step
		We multiply measurements from 3 original gaussian with 
		6 possible permutation matrices. Then we take the top 3 scoring
		out of the 18 generated.
		"""
		# predict 
		for i in range(self.num_targets):
			pred_mu[i] = self.dynamics(pred_mu[i], t, add_noise=False)
			pred_cov[i] = np.eye(6).dot(pred_cov[i]).dot( np.eye(6).T ) + self.Q

		perm_mats = self.generate_permutation_matrices()
		alphas = []
		hypothesis_idx_vec = []
		# loop over hypotheses
		for j in range(self.num_targets):
			# loop over measurement associations
			for perm_mat in perm_mats:
				# this is the expected measurement
				gauss_mu = perm_mat.dot(pred_mu[j,:])
				gauss_cov = self.R + perm_mat.dot(pred_cov[j]).dot(perm_mat.T)
				# these are the scores alpha_{ij}
				# # these scores are the probabilities p(y_t | y_{1:t-1} )
				alpha = self.weights[j] * self.gauss_pdf(x=y_true, mu=gauss_mu,cov=gauss_cov)
				alphas += [alpha]
				hypothesis_idx_vec += [j]

		# prune from 18 to 3
		alphas = np.array(alphas)
		top3scores = np.argsort(alphas)
		top3_score_idxs = top3scores[::-1][0:3] # reverse, then take top 3 scores

		# update each hypothesis from among 18 choices
		for i in range(3):
			j = top3_score_idxs[i]
			perm_mat = perm_mats[j % 6]
			self.weights[i] = alphas[j]
			hypothesis_idx = hypothesis_idx_vec[j] # this is the trajectory we associated w it 

			inv = perm_mat.dot( pred_cov[hypothesis_idx]).dot(perm_mat.T) + self.R
			K = pred_cov[hypothesis_idx].dot(perm_mat.T) * scipy.linalg.inv(inv)
			pred_meas = perm_mat.dot(pred_mu[hypothesis_idx])
			pred_mu[i] = pred_mu[hypothesis_idx] + K.dot( y_true - pred_meas)
			pred_cov[i] = (np.eye(6) - K.dot(perm_mat)).dot(pred_cov[hypothesis_idx])

		self.weights = self.weights / np.sum(self.weights)
		return pred_mu, pred_cov


def run_MHKF_filter():
	num_timesteps = 50
	mhkf = MultiHypothesisKalmanFilter(num_timesteps)
	true_states, true_meas = mhkf.generate_sequence()

	pred_states = []

	pred_state, pred_cov = mhkf.get_initial_state_estimate()
	for t in range(num_timesteps):
		y_true = true_meas[t]
		pred_state, pred_cov = mhkf.run_mhkf_step(pred_state, pred_cov, t, y_true)
		pred_states += [pred_state]

	pred_states = np.array(pred_states)
	true_states = np.array(true_states)

	# Keep around 3 possible hypothesis of assignments
	plt.plot(true_states[:,0],true_states[:,1], 'r', linestyle='solid')
	plt.plot(true_states[:,2],true_states[:,3], 'g', linestyle='solid')
	plt.plot(true_states[:,4],true_states[:,5], 'b', linestyle='solid')
	# plt.savefig('true.png')

	# Hypothesis 1
	plt.plot(pred_states[:,1,0], pred_states[:,1,1], 'r', linestyle='dashed')
	plt.plot(pred_states[:,1,2], pred_states[:,1,3], 'g', linestyle='dashed')
	plt.plot(pred_states[:,1,4], pred_states[:,1,5], 'b', linestyle='dashed')
	# plt.savefig('pred.png')
	plt.show()