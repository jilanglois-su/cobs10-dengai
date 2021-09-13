import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp


class HMM:
    def __init__(self, num_states):
        self.num_states = num_states

    @staticmethod
    def forward_pass(initial_dist, transition_matrix, log_likelihoods):
        """Perform the forward pass and return the forward messages for
        a single "event".

        In the descriptions below, let K denote the number of discrete states
        and T the number of time steps.

        Parameters
        ---
        initial_dist: (K,) array with initial state probabilities
        transition_matrix: (K, K) array where each row is a transition probability
        log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)

        Returns
        ---
        alphas: (T, K) array of forward messages
        """
        T, K = log_likelihoods.shape
        log_alphas = np.zeros((T, K))
        log_alphas[0, :] = np.log(initial_dist)
        for t in range(1, T):
            factor = log_alphas[t-1, :] + log_likelihoods[t-1, :]
            log_alphas_next = logsumexp(np.log(transition_matrix) + factor[:, np.newaxis], axis=0)
            log_alphas[t, :] = log_alphas_next - logsumexp(factor)[np.newaxis]

        return log_alphas

    @staticmethod
    def compute_marginal_ll(log_alphas, log_likelihoods):
        """Compute the marginal likelihood using the forward messages.

        Parameters
        ----------
        log_alphas: (T, K) array of forward messages.
        log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)

        Returns
        -------
        marginal_ll: real-valued scalar, log p(x_{1:T})
        """

        return logsumexp(log_alphas + log_likelihoods, axis=1).sum()

    @staticmethod
    def backward_pass(transition_matrix, log_likelihoods):
        """Perform the backward pass and return the backward messages for
        a single "event".

        Parameters
        ---
        transition_matrix: (K, K) array where each row is a transition probability
        log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)

        Returns
        ---
        log_betas: (T, K) array of backward messages
        """

        T, K = log_likelihoods.shape
        log_betas = np.zeros((T, K))
        for t in range(1, T):
            factor = log_betas[T-t, :] + log_likelihoods[T-t, :]
            log_betas_next = logsumexp(np.log(transition_matrix) + factor[np.newaxis, :], axis=1)
            log_betas[T-1-t, :] = log_betas_next

        return log_betas

    @staticmethod
    def compute_log_likelihoods(data, means, covariances):
        """Perform the backward pass and return the backward messages for
          a single "event".

          Parameters
          ---
          data: (T, 20) array with player positions over time for a particular even
          means = (K, 20) array with the means b_k
          covariances = (K, 20, 20) array with the covariances Q_k

          Returns
          ---
          log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)
          """
        T = data.shape[0]
        p = data.shape[1]
        K = means.shape[0]
        log_likelihoods = np.zeros((T, K))
        for k in range(K):
            icov_k = np.linalg.inv(covariances[k, :, :])
            (sign, logdet) = np.linalg.slogdet(icov_k)
            x = data.reshape((T, p, 1)) - means[k, :][np.newaxis, :, np.newaxis]

            quadratic_term = np.einsum('...ji,jk,...kl', x, icov_k, x).flatten()

            log_likelihoods[:, k] = - 0.5 * quadratic_term + 0.5 * sign * logdet - 0.5 * p * np.log(2 * np.pi)

        return log_likelihoods

    def e_step(self, event_data, parameters):
        """Run the E step for each event First compute the log likelihoods
        for each time step and discrete state using the given data and parameters.
        Then run the forward and backward passes and use the output to compute the
        posterior marginals, and use marginal_ll to compute the marginal likelihood.

        Parameters
        ---
        event_data: list of (T, 20) arrays with player positions over time for each event
        parameters: a data structure containing the model parameters; i.e. the
            initial distribution, transition matrix, and Gaussian means and
            covariances.

        Returns
        ---
        expectations: list of (T, K) arrays of marginal probabilities
            p(z_t = k | x_{1:T}) for each event.
        marginal_ll: marginal log probability p(x_{1:T}). This should go up
            each iteration!
        """
        initial_dist = parameters['initial_dist']
        transition_matrix = parameters['transition_matrix']
        means = parameters['means']
        covariances = parameters['covariances']

        expectations = []
        marginal_ll = 0
        for i in range(len(event_data)):
            log_likelihoods = self.compute_log_likelihoods(event_data[i], means, covariances)
            ll_check = log_likelihoods.sum(axis=0) > 0
            if ll_check.any():
                p = means.shape[1]
                new_covariance = self.initialize_covariance(p)
                covariances[ll_check, :, :] = new_covariance[ll_check, :, :]
                log_likelihoods = self.compute_log_likelihoods(event_data[i], means, covariances)

            log_alphas = self.forward_pass(initial_dist, transition_matrix, log_likelihoods)
            log_betas = self.backward_pass(transition_matrix, log_likelihoods)

            log_expectations_batch = log_alphas + log_likelihoods + log_betas
            log_expectations_batch = log_expectations_batch - logsumexp(log_expectations_batch, axis=1)[:, np.newaxis]

            expectations += [np.exp(log_expectations_batch)]
            marginal_ll += self.compute_marginal_ll(log_alphas=log_alphas, log_likelihoods=log_likelihoods)

        return expectations, marginal_ll

    def m_step(self, data, expectations):
        """Solve for the Gaussian parameters that maximize the expected log
        likelihood.

        Note: you can assume fixed initial distribution and transition matrix as
        described in the markdown above.

        Parameters
        ----------
        data: list of (T, 20) arrays with player positions over time for each event
        expectations: list of (T, K) arrays with marginal state probabilities from
            the E step.

        Returns
        -------
        parameters: a data structure containing the model parameters; i.e. the
            initial distribution, transition matrix, and Gaussian means and
            covariances.
        """
        K = expectations[0].shape[1]
        data = np.vstack(data)
        expectations = np.vstack(expectations)
        psudo_counts = expectations.sum(axis=0)
        means = []
        covariances = []
        for k in range(K):
            t1 = np.dot(data.T, expectations[:, k][:, np.newaxis] * data)
            t2 = (expectations[:, k][:, np.newaxis] * data).sum(axis=0)
            t3 = psudo_counts[k]
            mean_k = t2/t3
            tt2 = np.einsum('i,j', t2.flatten(), t2.flatten())
            covariance_k = (t1 - tt2/t3) / psudo_counts[k]
            means += [mean_k]
            covariances += [covariance_k]

        parameters = {'means': np.array(means), 'covariances': np.array(covariances),
                      'initial_dist': psudo_counts / psudo_counts.sum(),
                      'transition_matrix': self.initial_transition_matrix(K)}

        return parameters

    def fit(self, event_data):
        """Fit an HMM using the EM algorithm above. You'll have to initialize the
        parameters somehow; k-means often works well. You'll also need to monitor
        the marginal likelihood and check for convergence.

        Returns
        -------
        lls: the marginal log likelihood over EM iterations
        parameters: the final parameters
        """

        parameters = self.initialization(event_data)
        lls = []
        improvement = 10
        c = 0
        print("Solving", end="", flush=True)
        while improvement > 1:
            expectations, marginal_ll = self.e_step(event_data, parameters)
            parameters = self.m_step(event_data, expectations)

            if len(lls) > 0:
                improvement = marginal_ll - lls[-1]
                lls += [marginal_ll]
            else:
                lls += [marginal_ll]
            print(".", end="", flush=True)
            c += 1
            if c > 50:
                break
        print("Done")
        return lls, parameters

    @staticmethod
    def initial_transition_matrix(K, e=0.05):
        transition_matrix = np.ones((K, K)) * e / (K - 1)
        np.fill_diagonal(transition_matrix, 1. - e)

        return transition_matrix

    def initialization(self, event_data, kmeans_cov=False):
        parameters = dict()
        data = np.vstack(event_data)
        p = data.shape[1]
        kmeans = KMeans(n_clusters=self.num_states).fit(data)
        parameters['means'] = kmeans.cluster_centers_
        parameters['covariances'] = self.initialize_covariance(p)
        if kmeans_cov:
            latent_states = kmeans.predict(data)
            for state in range(self.num_states):
                parameters['covariances'][state, :, :] = np.cov(data[latent_states == state, :].T)
        parameters['transition_matrix'] = self.initial_transition_matrix(K=self.num_states)
        parameters['initial_dist'] = np.ones(self.num_states) / self.num_states

        return parameters

    def initialize_covariance(self, p):
        return np.tile(625 * np.eye(p), (self.num_states, 1, 1))
