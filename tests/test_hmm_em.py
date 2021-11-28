from unittest import TestCase
import numpy as np
from scipy.special import logsumexp
from src.d04_modeling.poisson_hmm import HMM
from scipy.stats import poisson, multivariate_normal


class TestHmmEm(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("setUp")

        cls.num_states = 2
        cls.model = HMM(num_states=cls.num_states)
        cls.p = 3

        cls.pi = np.ones(cls.num_states)
        np.random.seed(1992)
        mu1 = np.random.normal(loc=0.0, scale=1., size=cls.p)
        mu2 = np.random.normal(loc=0.0, scale=1., size=cls.p)
        cls.mu = np.array([mu1, mu2])
        cls.transition_matrix = np.random.dirichlet([1.] * cls.num_states, size=cls.num_states)
        # generate data
        cls.num_periods = 1000
        cls.generate_data(cls.num_periods)

        cls.initial_dist = np.ones(cls.num_states) / cls.num_states

    @classmethod
    def generate_data(cls, num_periods):
        x_data = np.ones((num_periods, cls.p))
        x_data[:, 1:] = multivariate_normal.rvs(mean=np.zeros(cls.p-1), cov=np.eye(cls.p-1), size=num_periods)
        y_data = np.zeros((num_periods,))
        latent_states = np.zeros(num_periods)
        rate = np.exp(np.dot(x_data[0, :], cls.mu[0]))
        y_data[0] = poisson.rvs(mu=rate)
        for t in range(1, num_periods):
            p = cls.transition_matrix[int(latent_states[t - 1]), :]
            z = np.random.choice(np.arange(cls.num_states).astype(int), p=p)
            rate = np.exp(np.dot(x_data[t, :], cls.mu[z]))
            y_data[t] = poisson.rvs(mu=rate)
            latent_states[t] = z
        cls.event_data = dict(x=[x_data], y=[y_data])
        cls.latent_states = latent_states

    def test_compute_log_likelihoods(self):
        log_likelihoods = self.model.compute_log_likelihoods(x_data=self.event_data['x'][0],
                                                             y_data=self.event_data['y'][0],
                                                             mu=self.mu, num_periods=self.num_periods)
        rate0 = np.exp(np.dot(self.event_data['x'][0], self.mu[0]))
        log_p0 = poisson.logpmf(k=self.event_data['y'][0], mu=rate0)
        rate1 = np.exp(np.dot(self.event_data['x'][0], self.mu[1]))
        log_p1 = poisson.logpmf(k=self.event_data['y'][0], mu=rate1)

        self.assertAlmostEqual(log_likelihoods[:, 0].sum(), log_p0.sum())
        self.assertAlmostEqual(log_likelihoods[:, 1].sum(), log_p1.sum())

    def test_forward_pass(self):
        log_likelihoods = self.model.compute_log_likelihoods(x_data=self.event_data['x'][0],
                                                             y_data=self.event_data['y'][0],
                                                             mu=self.mu, num_periods=self.num_periods)
        log_alphas = self.model.forward_pass(initial_dist=self.initial_dist, transition_matrix=self.transition_matrix,
                                             log_likelihoods=log_likelihoods)

        num_periods, num_states = log_likelihoods.shape
        expected = np.zeros((num_periods, num_states))
        expected[0, :] = np.log(self.initial_dist)
        for t in range(1, num_periods):
            factor = expected[t-1, :] + log_likelihoods[t-1, :]
            for next_state in range(num_states):
                log_alphas_next = logsumexp(np.log(self.transition_matrix[:, next_state].flatten()) + factor)
                expected[t, next_state] = log_alphas_next
            normalizing_factor = expected[t-1, :] + log_likelihoods[t-1, :]
            expected[t, :] = expected[t, :] - logsumexp(normalizing_factor, axis=0)[np.newaxis]

        self.assertEqual(0, (log_alphas - expected).sum())

    def test_backward_pass(self):
        log_likelihoods = self.model.compute_log_likelihoods(x_data=self.event_data['x'][0],
                                                             y_data=self.event_data['y'][0],
                                                             mu=self.mu, num_periods=self.num_periods)
        log_betas = self.model.backward_pass(transition_matrix=self.transition_matrix, log_likelihoods=log_likelihoods)

        num_periods, num_states = log_likelihoods.shape
        expected = np.zeros((num_periods, num_states))
        for t in range(1, num_periods):
            factor = expected[num_periods-t, :] + log_likelihoods[num_periods-t, :]
            for prev_state in range(num_states):
                log_betas_prev = logsumexp(np.log(self.transition_matrix[prev_state, :].flatten()) + factor)
                expected[num_periods-1-t, prev_state] = log_betas_prev

        self.assertEqual(0, (log_betas - expected).sum())

    def test_compute_marginal_ll(self):
        log_likelihoods = self.model.compute_log_likelihoods(x_data=self.event_data['x'][0],
                                                             y_data=self.event_data['y'][0],
                                                             mu=self.mu, num_periods=self.num_periods)
        log_alphas = self.model.forward_pass(initial_dist=self.initial_dist, transition_matrix=self.transition_matrix,
                                             log_likelihoods=log_likelihoods)
        marginal_ll = self.model.compute_marginal_ll(log_alphas, log_likelihoods)

        expected = logsumexp(log_likelihoods + log_alphas, axis=1)
        self.assertEqual(marginal_ll, expected.sum())

    def test_e_step(self):
        parameters = {'mu': self.mu,
                      'initial_dist': self.initial_dist,
                      'transition_matrix': self.transition_matrix}
        expectations, marginal_ll, transition_expectations = self.model.e_step(self.event_data, parameters)
        # expectations, transition_expectations = self.view_latent_states()

        self.assertAlmostEqual(first=expectations[0].sum(), second=self.num_periods, places=6)
        self.assertAlmostEqual(first=transition_expectations[0].sum(axis=0).sum(axis=0)[0], second=1, places=6)
        self.assertAlmostEqual(first=transition_expectations[0].sum(),
                               second=(self.num_periods-1), places=6)
        self.assertTrue(isinstance(marginal_ll, np.float64))

    def test_m_step(self):
        parameters = {'mu': self.mu,
                      'initial_dist': self.initial_dist,
                      'transition_matrix': self.transition_matrix}
        # expectations, marginal_ll, transition_expectations = self.model.e_step(self.event_data, parameters)
        expectations, transition_expectations = self.view_latent_states()
        parameters = self.model.m_step(self.event_data, expectations, transition_expectations)

        self.assertEqual(parameters['mu'].shape, (self.num_states, self.p))
        self.assertEqual(parameters['transition_matrix'].shape, (self.num_states, self.num_states))

    def view_latent_states(self):
        expectations = np.zeros((self.num_periods, self.num_states))
        transition_expectations = np.zeros((self.num_states, self.num_states, self.num_periods - 1))
        for k in range(self.num_states):
            mask = self.latent_states == k
            expectations[mask, k] = 1.
        expectations = [expectations]
        for i in range(self.num_states):
            for j in range(self.num_states):
                mask = (self.latent_states[1:] == j) & (self.latent_states[:-1] == i)
                transition_expectations[i, j, mask] = 1.
        transition_expectations = [transition_expectations]
        return expectations, transition_expectations

    def test_fit_hmm(self):
        lls, parameters = self.model.fit(self.event_data)

        lls = np.diff(lls)

        self.assertTrue((lls > 0).all())

