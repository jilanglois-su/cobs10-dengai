import numpy as np
import torch
from src.d00_utils.iohmm_utils import Likelihood, np2torch


class IOHMM:
    """
    An Input Output HMM Architecture, Bengio and Frasconi 1995
    """

    def __init__(self, num_features, num_states):
        self.num_states = num_states
        self.num_featuers = num_features
        self.likelihood = Likelihood(num_features, num_states)
        self.likelihood_optimizer = torch.optim.Adam(self.likelihood.parameters(),
                                                     lr=self.likelihood.lr)

    def e_step(self, event_data):
        """
        Run the E step for each event First compute the log likelihoods
        for each time step and discrete state using the given data and parameters.
        Then run the forward and backward passes and use the output to compute the
        posterior marginals, and use marginal_ll to compute the marginal likelihood.

        Parameters
        ---
        event_data: list of (input_observations (T, d), output_observation (T, 1)) arrays over
        time for each event.

        Returns
        ---
        expectations: list of (T, K) arrays of marginal probabilities
            p(z_t = k | x_{1:t}) for each event (filtered).

        transition_expectations: list of (T, K, K) arrays of expected transitions
        """

        expectations = []
        transition_expectations = []
        num_events = len(event_data)
        for p in range(num_events):
            input_observations, output_observations = event_data[p]
            num_periods = input_observations.shape[1]
            log_betas_event = np2torch(np.zeros((num_periods, self.num_states)))
            log_alphas_event = np2torch(np.zeros((num_periods+1, self.num_states)))
            log_alphas_event[0, :] = torch.log(self.likelihood.get_initial_dist())
            log_expectations_event = np2torch(np.zeros((num_periods+1, self.num_states)))
            log_expectations_event[0, :] = np2torch(np.log(np.ones(self.num_states) / self.num_states))
            # log_psi (t, j, i)
            log_psi = torch.squeeze(self.likelihood.get_log_psi(input_observations))
            obs_distribution = self.likelihood.get_distribution(input_observations)
            poisson_log_prob = torch.squeeze(obs_distribution.log_prob(output_observations))
            for t in range(1, num_periods+1):
                self.backward_step(log_betas_event, log_psi, poisson_log_prob, num_periods-t)
                self.forward_step(log_alphas_event, log_psi, poisson_log_prob, t)

                log_expectations_term = torch.zeros(log_psi[t-1].shape) + log_expectations_event[t-1, :, None]
                log_expectations_event[t, :] = torch.logsumexp(log_expectations_term + log_psi[t-1], dim=0)

            log_beta_term = torch.zeros(log_psi[:, :, :].shape) + log_betas_event[:, None, :]
            log_alpha_term = torch.zeros(log_psi[:, :, :].shape) + log_alphas_event[:-1, :, None]
            log_transition_expectations_event = log_beta_term + log_alpha_term + log_psi
            # TODO: looks weird that normalization is only with last value of alpha
            normalization_term = torch.logsumexp(log_alphas_event[-1, :], dim=-1)
            log_transition_expectations_event -= normalization_term

            expectations += [torch.exp(log_expectations_event[1:])]
            transition_expectations += [torch.exp(log_transition_expectations_event)]

        return expectations, transition_expectations

    def backward_step(self, log_betas_event, log_psi, poisson_log_prob, t):
        log_beta_term = torch.zeros(log_psi[t].shape) + log_betas_event[t, None, :]
        log_sum1 = torch.logsumexp(log_psi[t] + log_beta_term, dim=1)
        log_betas_event[t - 1, :] = poisson_log_prob[t, :] + log_sum1

    def forward_step(self, log_alphas_event, log_psi, poisson_log_prob, t):
        log_alpha_term = torch.zeros(log_psi[t-1].shape) + log_alphas_event[t - 1, :, None]
        log_sum0 = torch.logsumexp(log_psi[t-1] + log_alpha_term, dim=0)
        log_alphas_event[t, :] = poisson_log_prob[t-1, :] + log_sum0

    def m_step(self, event_data, expectations, transition_expectations):
        """Solve for the Gaussian parameters that maximize the expected log
        likelihood.

        Note: you can assume fixed initial distribution and transition matrix as
        described in the markdown above.

        Parameters
        ---
        event_data: list of (input_observations (T, d), output_observation (T, 1)) arrays over
        time for each event.
        expectations: list of (T, K) arrays of marginal probabilities
            p(z_t = k | x_{1:t}) for each event (filtered).
        transition_expectations: list of (T, K, K) arrays of expected transitions

        Returns
        -------
        loglikelihood
        """

        lls = None
        for i in range(10):
            self.likelihood_optimizer.zero_grad()
            loss = -self.likelihood.get_log_likelihood(event_data, transition_expectations, expectations)
            loss.backward()
            self.likelihood_optimizer.step()
            lls = loss.item()

        return lls

    def fit(self, event_data, num_iterations=100):
        """Fit an HMM using the EM algorithm above. You'll have to initialize the
        parameters somehow; k-means often works well. You'll also need to monitor
        the marginal likelihood and check for convergence.

        Returns
        -------
        lls: the marginal log likelihood over EM iterations
        parameters: the final parameters
        """
        lls = []
        c = 0
        # print("Solving", end="", flush=True)
        print("Solving")
        while c < num_iterations:
            # E-Step
            expectations, transition_expectations = self.e_step(event_data)
            # M-Step
            lls_iter = self.m_step(event_data, expectations, transition_expectations)

            lls += [lls_iter]
            print("[iter %3i] Loss: %10.4f" % (c, lls_iter))

        print("Done")
        return lls

    @staticmethod
    def format_event_data(x_data, y_data):
        x_data.sort_index(inplace=True)
        y_data.sort_index(inplace=True)
        event_data = []
        for city in x_data.index.get_level_values('city').unique():
            if 'year' in x_data.index.names:
                for year in x_data.loc[city].index.get_level_values('year').unique():
                    num_periods = len(x_data.loc[city].loc[year])
                    input_observations = x_data.loc[city].loc[year].values.reshape(1, num_periods, -1)
                    output_observations = y_data.loc[city].loc[year].values.reshape(1, num_periods, -1)
                    event_data.append((np2torch(input_observations), np2torch(output_observations)))
            else:
                num_periods = len(x_data.loc[city])
                input_observations = x_data.loc[city].values.reshape(1, num_periods, -1)
                output_observations = y_data.loc[city].values.reshape(1, num_periods, -1)
                event_data.append((np2torch(input_observations), np2torch(output_observations)))
        return event_data


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    import matplotlib.pyplot as plt
    os.chdir('../')
    dda = DengueDataApi()
    x_train, x_validate, y_train, y_validate = dda.split_data()
    z_train, z_validate, pct_var = dda.get_pca(x_train, x_validate, num_components=4)
    model = IOHMM(num_states=3, num_features=z_train.shape[1])
    event_data = model.format_event_data(z_train, y_train)
    lls = model.fit(event_data=event_data)

    plt.plot(lls)
    plt.show()



