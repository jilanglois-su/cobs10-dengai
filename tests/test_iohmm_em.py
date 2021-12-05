from unittest import TestCase
import torch
from src.d04_modeling.iohmm import IOHMM
from src.d01_data.dengue_data_api import DengueDataApi


class TestHmmEm(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("setUp")
        num_states = 3
        dda = DengueDataApi()
        x_train, x_validate, y_train, y_validate = dda.split_data()
        z_train, z_validate, pct_var = dda.get_pca(x_train, x_validate, num_components=4)
        cls.num_states = num_states
        cls.model = IOHMM(num_states=num_states, num_features=z_train.shape[1])
        cls.event_data = cls.model.format_event_data(z_train.droplevel('year'), y_train.droplevel('year'))

    def test_e_step(self):
        expectations, transition_expectations, log_alphas, log_betas = self.model.e_step(self.event_data)
        expectations_, transition_expectations_, log_alphas_, log_betas_ = self.slow_e_step(self.event_data)
        for p in range(len(self.event_data)):
            num_periods = self.event_data[p][0].shape[0]
            test_value1 = expectations[p].sum().sum().item()
            test_value2 = transition_expectations[p].sum().sum().sum().item()
            self.assertAlmostEqual(num_periods, test_value1, places=0)
            self.assertAlmostEqual(num_periods, test_value2, places=0)

            diff_expectations = (expectations[p] - expectations_[p]).sum(dim=0)
            diff_transition = (transition_expectations[p] - transition_expectations_[p]).sum(dim=1)
            diff_alphas = (log_alphas[p] - log_alphas_[p]).sum(dim=0)
            diff_betas = (log_betas[p] - log_betas_[p]).sum(dim=0)

            self.assertAlmostEqual(diff_expectations.sum(), 0, places=0)
            self.assertAlmostEqual(diff_transition.sum().sum(), 0, places=0)
            self.assertAlmostEqual(diff_alphas.sum(), 0, places=0)
            self.assertAlmostEqual(diff_betas.sum(), 0, places=0)

    def slow_e_step(self, event_data):
        expectations = []
        transition_expectations = []
        log_alphas = []
        log_betas = []
        num_events = len(event_data)
        for p in range(num_events):
            input_observations, output_observations = event_data[p]
            num_periods = input_observations.shape[0]
            initial_eval = self.model.initialize_e_step(input_observations, num_periods, output_observations)
            log_alphas_event, log_betas_event, log_expectations_event, log_psi, poisson_log_prob = initial_eval
            for t in range(1, num_periods+1):
                for i in range(self.num_states):
                    log_sum1 = torch.logsumexp(log_psi[i, num_periods - t, :] + log_betas_event[num_periods - t], dim=0)
                    log_betas_event[num_periods - t - 1, i] = poisson_log_prob[i, num_periods - t] + log_sum1

                    log_sum0 = torch.logsumexp(log_psi[:, t - 1, i] + log_alphas_event[t - 1], dim=0)
                    log_alphas_event[t, i] = poisson_log_prob[i, t - 1] + log_sum0

                    log_expectations_event[t, i] = torch.logsumexp(log_expectations_event[t-1, :] + log_psi[:, t-1, i], dim=0)

            log_beta_term = torch.zeros(log_psi.shape) + log_betas_event[None, :, :]
            log_alpha_term = torch.zeros(log_psi.shape) + torch.transpose(log_alphas_event[:-1, :], 0, 1)[:, :, None]
            log_transition_expectations_event = log_beta_term + log_alpha_term + log_psi
            # TODO: check normalization from paper
            normalization_term = torch.logsumexp(torch.logsumexp(log_transition_expectations_event, dim=-1), dim=0)
            log_transition_expectations_event -= normalization_term[None, :, None]

            expectations += [torch.exp(log_expectations_event[1:])]
            transition_expectations += [torch.exp(log_transition_expectations_event)]
            log_alphas += [log_alphas_event]
            log_betas += [log_betas_event]

        return expectations, transition_expectations, log_alphas, log_betas

