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
        expectations, transition_expectations, log_alphas, log_betas, _ = self.model.e_step(self.event_data)
        expectations_, transition_expectations_, log_alphas_, log_betas_ = self.slow_e_step(self.event_data)
        for p in range(len(self.event_data)):
            num_periods = self.event_data[p][0].shape[0]
            expectations_event = expectations[p]
            transition_expectations_event = transition_expectations[p]
            test_value1 = expectations_event.sum().sum().item()
            test_value2 = transition_expectations_event.sum().sum().sum().item()
            self.assertAlmostEqual(num_periods, test_value1, places=0, msg="Expectations do not sum to 1!")
            self.assertAlmostEqual(num_periods-1, test_value2, places=0, msg="Transition Expectations do not sum to 1!")

            diff_expectations = (expectations[p] - expectations_[p]).sum(dim=0)
            diff_transition = (transition_expectations[p] - transition_expectations_[p]).sum(dim=1)
            diff_alphas = (log_alphas[p] - log_alphas_[p]).sum(dim=0)
            diff_betas = (log_betas[p] - log_betas_[p]).sum(dim=0)

            self.assertAlmostEqual(diff_expectations.sum().item(), 0, places=0, msg="Expectations are wrong")
            self.assertAlmostEqual(diff_transition.sum().sum().item(), 0, places=0, msg="Transition Expectations are wrong")
            self.assertAlmostEqual(diff_alphas.sum().item(), 0, places=0, msg="Forward step is wrong")
            self.assertAlmostEqual(diff_betas.sum().item(), 0, places=0, msg="Backward step is wrong")

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
            log_alphas_event, log_betas_event, log_psi, poisson_log_prob = initial_eval
            for t in range(1, num_periods):
                for i in range(self.num_states):
                    log_sum1 = torch.logsumexp(poisson_log_prob[:, num_periods - t] + log_psi[i, num_periods - t, :]
                                               + log_betas_event[num_periods - t], dim=0)
                    log_betas_event[num_periods - t - 1, i] = log_sum1

                    log_sum0 = torch.logsumexp(log_psi[:, t, i] + log_alphas_event[t - 1], dim=0)
                    log_alphas_event[t, i] = poisson_log_prob[i, t] + log_sum0

            log_expectations_event = log_alphas_event + log_betas_event
            # Normalize log expectations
            log_expectations_event -= torch.logsumexp(log_expectations_event, dim=-1)[:, None]

            log_transition_expectations_event = torch.zeros(log_psi[:, 1:, :].shape)
            for i in range(self.num_states):
                for j in range(self.num_states):
                    log_transition_expectations_event[j, :, i] = log_alphas_event[:-1, j] + poisson_log_prob[i, 1:] \
                                                                 + log_psi[j, 1:, i] + log_betas_event[1:, i]
            # TODO: check normalization from paper
            normalization_term = torch.logsumexp(torch.logsumexp(log_transition_expectations_event, dim=-1), dim=0)
            log_transition_expectations_event -= normalization_term[None, :, None]

            expectations += [torch.exp(log_expectations_event)]
            transition_expectations += [torch.exp(log_transition_expectations_event)]
            log_alphas += [log_alphas_event]
            log_betas += [log_betas_event]

        return expectations, transition_expectations, log_alphas, log_betas

