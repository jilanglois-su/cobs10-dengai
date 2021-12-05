import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln
from scipy.stats import poisson
from src.d04_modeling.poisson_glm import PoissonGLM
import multiprocessing as mp

cpu_count = mp.cpu_count()
eps = 1.e-6


class PoissonHMM:
    def __init__(self, num_states, alpha=1., sigma2=1., seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.num_states = num_states
        if isinstance(alpha, float):
            alpha = np.ones(self.num_states) * alpha
        self.alpha = alpha
        self.sigma2 = sigma2

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

    @staticmethod
    def forward_pass(initial_dist, transition_matrix, log_likelihoods):
        """Perform the forward pass and return the forward messages for
        a single "event".

        In the descriptions below, let K denote the number of discrete states
        and T the number of time steps.

        transition_matrix as defined by STATS 271 lecture: P_{ij}=P(z_t=j \mid z_{t-1}=i)

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
            log_alphas_next = logsumexp(np.log(transition_matrix + eps) + factor[:, np.newaxis], axis=0)
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
            log_betas_next = logsumexp(np.log(transition_matrix + eps) + factor[np.newaxis, :], axis=1)
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
                                                                + np.log(transition_matrix[i, j] + eps) \
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

    def viterbi(self, event_data, parameters):
        initial_dist = parameters['initial_dist']
        transition_matrix = parameters['transition_matrix']
        mu = parameters['mu']

        most_likely_states = []
        for i in range(len(event_data['x'])):
            x_data = event_data['x'][i]
            y_data = event_data['y'][i]
            num_periods = x_data.shape[0]
            log_likelihoods = self.compute_log_likelihoods(x_data, y_data, mu, num_periods)
            ll_check = log_likelihoods.sum(axis=0) > 0
            if ll_check.any():
                raise Exception("Positive loglikelihoods!")

            T, K = log_likelihoods.shape
            log_mu = np.zeros((T, K))
            for t in range(1, T):
                factor = log_mu[T-t, :] + log_likelihoods[T-t, :]
                log_mu_next = np.max(np.log(transition_matrix + eps) + factor[np.newaxis, :], axis=1)
                log_mu[T-1-t, :] = log_mu_next

            most_likely_states_batch = [None] * T
            factor = log_likelihoods[0, :] + log_mu[0]
            most_likely_states_batch[0] = np.argmax(factor + np.log(initial_dist))
            for t in range(1, T):
                factor = log_likelihoods[t, :] + log_mu[t]
                prev_state = most_likely_states_batch[t-1]
                log_transition = np.log(transition_matrix[prev_state, :] + eps)
                most_likely_states_batch[t] = np.argmax(factor + log_transition)

            most_likely_states += [most_likely_states_batch]

        return most_likely_states

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
        prev_parameters = None
        while improvement > -1e-4:
            prev_parameters = parameters
            expectations, marginal_ll, transition_expectations = self.e_step(event_data, prev_parameters)
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
        return lls, prev_parameters

    def predict(self, event_data, parameters):
        expectations, marginal_ll, _ = self.e_step(event_data, parameters)
        most_likely_states = self.viterbi(event_data=event_data, parameters=parameters)
        y_viterbi = []
        for i in range(len(event_data['x'])):
            y_data = event_data['y'][i]
            x_data = event_data['x'][i]
            y_hat_event = np.zeros((y_data.shape[0], self.num_states))
            y_viterbi_event = np.zeros((y_data.shape[0], self.num_states))
            for k in range(self.num_states):
                mask = np.array(most_likely_states[i]) == k
                poisson_glm = PoissonGLM(x_train=x_data, y_train=y_data,
                                         sigma2=self.sigma2, bias=False)

                y_hat_event[:, k] = poisson_glm.obs_map(parameters['mu'][k], x_data).reshape(-1,)
                y_viterbi_event[mask, k] = 1.
            y_viterbi += [np.sum(y_hat_event * y_viterbi_event, axis=1)]

        return y_viterbi, most_likely_states, marginal_ll

    def forecast(self, train_event_data, test_event_data, parameters, m=8, num_samples=250, alpha=0.05):
        initial_dist = parameters['initial_dist']
        transition_matrix = parameters['transition_matrix']
        mu = parameters['mu']
        state_space = list(range(self.num_states))
        forecasts = []
        states_prob = []
        for i in range(len(train_event_data['x'])):
            x_train = train_event_data['x'][i]
            y_train = train_event_data['y'][i]
            poisson_glm = PoissonGLM(x_train=x_train, y_train=y_train,
                                     sigma2=self.sigma2, bias=False)

            x_test = test_event_data['x'][i]
            y_test = test_event_data['y'][i]
            num_periods = x_train.shape[0]
            test_periods = x_test.shape[0]
            forecasts_event = pd.DataFrame(np.nan, index=np.arange(test_periods), columns=['map', 'lower', 'upper'])
            states_prob_event = pd.DataFrame(np.nan, index=np.arange(test_periods), columns=state_space)
            x_data = x_train.copy()
            y_data = y_train.copy()
            print("Sampling", end="", flush=True)
            for t in range(x_test.shape[0]-m):
                log_likelihoods = self.compute_log_likelihoods(x_data, y_data, mu, num_periods)
                log_alphas = self.forward_pass(initial_dist, transition_matrix, log_likelihoods)
                log_filter_prob = log_alphas - logsumexp(log_alphas, axis=1)[:, np.newaxis]
                initial_dist = np.exp(log_filter_prob[-1, :])
                m_step_dist = np.dot(np.linalg.matrix_power(transition_matrix.T, m), initial_dist)
                states_prob_event.at[t+m] = m_step_dist
                states_sim = np.random.choice(state_space, size=num_samples, p=m_step_dist)
                mu_sim = mu[list(states_sim)]
                rate_sim = poisson_glm.rate_map(mu_sim.T, x_test[t + m - 1, :])
                obs_sim = poisson.rvs(rate_sim).flatten()
                lower_value = np.percentile(obs_sim, q=100*alpha/2)
                upper_value = np.percentile(obs_sim, q=100*(1.-alpha/2))
                map_value = np.median(obs_sim)
                forecasts_event.at[t+m, 'lower'] = lower_value
                forecasts_event.at[t+m, 'upper'] = upper_value
                forecasts_event.at[t+m, 'map'] = map_value
                if t % 10 == 0:
                    print(".", end="", flush=True)
                num_periods += 1
                x_data = np.vstack([x_data, x_test[t, :].reshape(1, -1)])
                y_data = np.append(y_data, y_test[t])

            print("Done")
            forecasts += [forecasts_event]
            states_prob += [states_prob_event]

        return forecasts, states_prob

    @staticmethod
    def format_event_data(df):
        df.sort_index(inplace=True)
        event_data = []
        for city in df.index.get_level_values('city').unique():
            if 'year' in df.index.names:
                for year in df.loc[city].index.get_level_values('year').unique():
                    event_data.append(df.loc[city].loc[year].values)
            else:
                event_data.append(df.loc[city].values)
        return event_data

    def validate_model(self, event_data, parameters):
        mu = parameters['mu']
        expectations, marginal_ll, _ = self.e_step(event_data, parameters)
        expectations, x_data, y_data = self.glm_inputs_setup(event_data, expectations)
        y_hat = np.zeros(y_data.shape)
        for k in range(self.num_states):
            poisson_glm = PoissonGLM(x_train=x_data, y_train=y_data, weights=expectations[:, k].reshape((-1, 1)),
                                     sigma2=self.sigma2, bias=False)

            y_hat += poisson_glm.obs_map(mu[k], x_data) * expectations[:, k].reshape((-1, 1))
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
    num_components = 4
    z_train, z_validate, pct_var = dda.get_pca(x_train, x_validate, num_components=num_components)
    z_train['bias'] = 1.
    z_validate['bias'] = 1.
    z_train.drop(columns=z_train.columns[:num_components], inplace=True)
    z_validate.drop(columns=z_validate.columns[:num_components], inplace=True)
    event_data_train = dict()
    model = PoissonHMM(num_states=3)
    event_data_train['x'] = model.format_event_data(z_train.droplevel('year'))
    event_data_train['y'] = model.format_event_data(y_train.droplevel('year'))
    lls_k, parameters_k = model.fit(event_data=event_data_train)
    print(lls_k)
    print(parameters_k)

    event_data_validate = dict()
    event_data_validate['x'] = model.format_event_data(z_validate.droplevel('year'))
    event_data_validate['y'] = model.format_event_data(y_validate.droplevel('year'))

    y_viterbi_train, most_likely_states_train, _ = model.predict(event_data_train, parameters_k)
    y_viterbi_validate, most_likely_states_validate, _ = model.predict(event_data_validate, parameters_k)
    forecasts = model.forecast(event_data_train, event_data_validate, parameters_k)

    # marginal_ll, mae = model.validate_model(event_data=event_data_validate, parameters=parameters_k)

    # print(mae)

