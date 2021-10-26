import numpy as np
from scipy.special import logsumexp
from scipy.special import gammaln
from src.d04_modeling.poisson_glm import PoissonGLM


class HMM:
    def __init__(self, num_states, alpha=1., sigma2=1.):
        self.num_states = num_states
        if isinstance(alpha, float):
            alpha = np.ones(self.num_states) * alpha
        self.alpha = alpha
        self.sigma2 = sigma2

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

    def compute_log_likelihoods(self, x_data, y_data, mu, num_periods):
        """Compute the log likelihood for a single "event".

          Parameters
          ---
          x_data: (T, p) array with features over time for a particular year
          y_data: (T, 1) array with counts over time for a particular year
          mu: (K, p) array with the Poisson GLM coefficients
          num_periods: T

          Returns
          ---
          log_likelihoods: (T, K) array with entries log p(y_t | x_t, mu, z_t=k)
          """
        log_likelihoods = np.zeros((num_periods, self.num_states))
        for k in range(self.num_states):
            log_rate_k = np.dot(x_data, mu[k])
            log_likelihoods[:, k] = y_data * log_rate_k - np.exp(log_rate_k) - gammaln(y_data+1)

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
        mu = parameters['mu']

        expectations = []
        transition_expectations = []
        marginal_ll = 0
        for i in range(len(event_data['x'])):
            x_data = event_data['x'][i]
            y_data = event_data['y'][i]
            num_periods = x_data.shape[0]
            log_likelihoods = self.compute_log_likelihoods(x_data, y_data, mu, num_periods)
            ll_check = log_likelihoods.sum(axis=0) > 0
            if ll_check.any():
                raise Exception("Positive loglikelihoods!")

            log_alphas = self.forward_pass(initial_dist, transition_matrix, log_likelihoods)
            log_betas = self.backward_pass(transition_matrix, log_likelihoods)

            log_expectations_batch = log_alphas + log_likelihoods + log_betas
            log_expectations_batch = log_expectations_batch - logsumexp(log_expectations_batch, axis=1)[:, np.newaxis]

            log_transition_expectation_batch = np.zeros(shape=[self.num_states, self.num_states, num_periods-1])
            for i in range(self.num_states):
                for j in range(self.num_states):
                    log_alphas_i = log_alphas[:-1, i]
                    log_likelihoods_i = log_likelihoods[:-1, i]
                    log_likelihoods_j = log_likelihoods[1:, j]
                    log_betas_j = log_betas[1:, j]
                    log_transition_expectation_batch[i, j, :] = log_alphas_i + log_likelihoods_i \
                                                                + np.log(transition_matrix[i, j]) \
                                                                + log_likelihoods_j + log_betas_j

            log_transition_expectation_batch = log_transition_expectation_batch \
                                               - logsumexp(log_transition_expectation_batch.reshape((-1, num_periods-1)), axis=0)[np.newaxis, np.newaxis, :]

            expectations += [np.exp(log_expectations_batch)]
            transition_expectations += [np.exp(log_transition_expectation_batch)]
            marginal_ll += self.compute_marginal_ll(log_alphas=log_alphas, log_likelihoods=log_likelihoods)

        return expectations, marginal_ll, transition_expectations

    def m_step(self, event_data, expectations, transition_expectations):
        """Solve for the Gaussian parameters that maximize the expected log
        likelihood.

        Note: you can assume fixed initial distribution and transition matrix as
        described in the markdown above.

        Parameters
        ----------
        event_data: list of (T, 20) arrays with player positions over time for each event
        expectations: list of (T, K) arrays with marginal state probabilities from
            the E step.
        transition_expectations: list of (K, K, T) arrays with marginal state transition
        probabilities from the E step

        Returns
        -------
        parameters: a data structure containing the model parameters; i.e. the
            initial distribution, transition matrix, and Gaussian means and
            covariances.
        """

        expectations, x_data, y_data = self.glm_inputs_setup(event_data, expectations)
        transition_expectations = np.concatenate(transition_expectations, axis=-1)
        psudo_counts = expectations.sum(axis=0)
        mu = []
        for k in range(self.num_states):
            poisson_glm = PoissonGLM(x_train=x_data, y_train=y_data, weights=expectations[:, k].reshape((-1, 1)),
                                     sigma2=self.sigma2, bias=False)
            poisson_glm.compute_posterior_mode()
            mu += [poisson_glm.get_w_map()]

        transition_matrix = np.zeros(shape=[self.num_states] * 2)
        for i in range(self.num_states):
            for j in range(self.num_states):
                transition_matrix[i, j] = transition_expectations[i, j, :].sum()
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis]

        parameters = {'mu': np.array(mu),
                      'initial_dist': psudo_counts / psudo_counts.sum(),
                      'transition_matrix': transition_matrix}

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

        p = event_data['x'][0].shape[1]
        parameters = self.initialization(p=p)
        lls = []
        improvement = 10
        c = 0
        print("Solving", end="", flush=True)
        while improvement > 1:
            expectations, marginal_ll, transition_expectations = self.e_step(event_data, parameters)
            parameters = self.m_step(event_data, expectations, transition_expectations)

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

    def initialization(self, p):
        parameters = dict()
        parameters['mu'] = np.random.normal(loc=0.0, scale=np.sqrt(self.sigma2), size=(self.num_states, p))
        transition_matrix = self.initial_transition_matrix(self.num_states)
        parameters['transition_matrix'] = transition_matrix
        parameters['initial_dist'] = np.ones(self.num_states) / self.num_states

        return parameters

    def initialize_covariance(self, p):
        return np.tile(625 * np.eye(p), (self.num_states, 1, 1))

    @staticmethod
    def format_event_data(df):
        df.sort_index(inplace=True)
        event_data = []
        for city in df.index.get_level_values('city').unique():
            for year in df.loc[city].index.get_level_values('year').unique():
                event_data.append(df.loc[city].loc[year].values)
        return event_data

    def validate_model(self, event_data, parameters):
        mu = parameters['mu']
        expectations, marginal_ll, _ = self.e_step(event_data, parameters)
        expectations, x_data, y_data = self.glm_inputs_setup(event_data, expectations)
        y_hat = np.zeros(y_data.shape)
        for k in range(self.num_states):
            poisson_glm = PoissonGLM(x_train=x_data, y_train=y_data, weights=expectations[:, k].reshape((-1, 1)),
                                     sigma2=self.sigma2, bias=False)

            y_hat += poisson_glm.obs_map(mu[k], x_validate) * expectations[:, k].reshape((-1, 1))
        e = np.abs(y_data - y_hat)
        mae = e.mean()

        return marginal_ll, mae

    def glm_inputs_setup(self, event_data, expectations):
        x_data = np.vstack([event_data['x'][i] for i in range(len(event_data['x']))])
        y_data = np.vstack([event_data['y'][i].reshape((-1, 1)) for i in range(len(event_data['y']))])
        expectations = np.vstack(expectations)
        return expectations, x_data, y_data


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    os.chdir('../')
    dda = DengueDataApi()
    x_train, x_validate, y_train, y_validate = dda.split_data()
    event_data = dict()
    model = HMM(num_states=3)
    event_data['x'] = model.format_event_data(x_train)
    event_data['y'] = model.format_event_data(y_train)
    lls_k, parameters_k = model.fit(event_data=event_data)
    print(lls_k)
    print(parameters_k)

    event_data_validate = dict()
    event_data_validate['x'] = model.format_event_data(x_validate)
    event_data_validate['y'] = model.format_event_data(y_validate)

    marginal_ll, mae = model.validate_model(event_data=event_data_validate, parameters=parameters_k)

    print(mae)

