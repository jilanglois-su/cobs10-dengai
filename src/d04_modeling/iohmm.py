import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import torch
import torch.nn.functional as F
from src.d00_utils.iohmm_utils import LinearWithChannel, np2torch
from src.d00_utils.constants import PATH_OUTPUT_NETWORK, PATH_STATE_NETWORK

torch.autograd.set_detect_anomaly(True)


class IOHMM:
    """
    An Input Output HMM Architecture, Bengio and Frasconi 1995
    """

    def __init__(self, num_features, num_states, learning_rate=0.001, load=True, model_name=None):
        if model_name is None:
            self.__model_name = 'iohmm_%i_%i' % (num_features, num_states)
        else:
            self.__model_name = model_name
        self.num_states = num_states
        self.num_featuers = num_features
        self.state_network = LinearWithChannel(input_size=num_features, output_size=num_states, channel_size=num_states)
        self.output_network = LinearWithChannel(input_size=num_features, output_size=1, channel_size=num_states)
        self.psudo_counts = torch.ones(self.num_states)
        self.state_optimizer = torch.optim.Adam(self.state_network.parameters(), lr=learning_rate)
        self.output_optimizer = torch.optim.Adam(self.output_network.parameters(), lr=learning_rate)
        if load:
            state_checkpoint = torch.load(PATH_STATE_NETWORK.format(name=self.__model_name))
            self.state_network.load_state_dict(state_checkpoint['model_state_dict'])
            self.psudo_counts = state_checkpoint['psudo_counts']
            self.state_optimizer.load_state_dict(state_checkpoint['optimizer_state_dict'])

            output_checkpoint = torch.load(PATH_OUTPUT_NETWORK.format(name=self.__model_name))
            self.output_network.load_state_dict(output_checkpoint['model_state_dict'])
            self.output_optimizer.load_state_dict(output_checkpoint['optimizer_state_dict'])

    def get_initial_dist(self):
        initial_dist = self.psudo_counts / self.psudo_counts.sum()
        return initial_dist

    def get_log_psi(self, input_observations):
        """
        :param input_observations:
        :return: torch.tensor(channel_size, batch_size, output_size)
        """
        intermediate_variables = self.state_network.forward(input_observations)
        log_psi = F.log_softmax(intermediate_variables, dim=-1)
        return log_psi

    def get_distribution(self, input_observations):
        """
        :param input_observations:
        :return: torch.tensor(channel_size, batch_size, output_size)
        """
        log_rates = self.output_network.forward(input_observations)
        distribution = torch.distributions.Poisson(rate=torch.exp(log_rates))
        return distribution

    def log_likelihood_emissions(self, input_observations, output_observations, expectations):
        """
        :param input_observations: torch.tensor(epoch_size, batch_size, output_size)
        :param output_observations: torch.tensor(epoch_size, batch_size, output_size)
        :param expectations: torch.tensor(epoch_size, batch_size, output_size)
        :return:
        """
        obs_distribution = self.get_distribution(input_observations)
        poisson_log_prob = torch.squeeze(obs_distribution.log_prob(output_observations))
        lls = (expectations * torch.transpose(poisson_log_prob, 0, 1)).sum().sum()

        return lls

    def log_likelihood_transitions(self, input_observations, transition_expectations):
        """
       :param input_observations: torch.tensor(batch_size, output_size)
       :param transition_expectations: torch.tensor(epoch_size, channel_size=num_states, batch_size, output_size=num_states)
       :return:
       """
        log_psi = torch.squeeze(self.get_log_psi(input_observations))
        lls = (transition_expectations * log_psi[:, 1:, :]).sum().sum().sum()

        return lls

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
        expectations: list of (batch_size, output_size=num_states) arrays of marginal probabilities
            p(z_t = k | x_{1:t}) for each event (filtered).

        transition_expectations: list of (channel_size=num_states, batch_size, output_size=num_states) arrays of
        expected transitions
        """

        expectations = []
        transition_expectations = []
        log_alphas = []
        log_betas = []
        num_events = len(event_data)
        for p in range(num_events):
            input_observations, output_observations = event_data[p]
            num_periods = input_observations.shape[0]
            initial_eval = self.initialize_e_step(input_observations, num_periods, output_observations)
            log_alphas_event, log_betas_event, log_psi, poisson_log_prob = initial_eval
            for t in range(1, num_periods):
                self.backward_step(log_betas_event, log_psi, poisson_log_prob, num_periods-t)
                self.forward_step(log_alphas_event, log_psi, poisson_log_prob, t)

            log_expectations_event = log_alphas_event + log_betas_event
            # Normalize log expectations
            log_expectations_event -= torch.logsumexp(log_expectations_event, dim=-1)[:, None]
            log_transition_expectations_event = log_psi[:, 1:, :]
            log_transition_expectations_event += torch.transpose(log_alphas_event[:-1, :], 0, 1)[:, :, None]
            log_transition_expectations_event += torch.transpose(poisson_log_prob[:, 1:], 0, 1)[None, :, :]
            log_transition_expectations_event += log_betas_event[1:, :][None, :, :]
            # TODO: check normalization from paper
            normalization_term = torch.logsumexp(torch.logsumexp(log_transition_expectations_event, dim=-1), dim=0)
            log_transition_expectations_event -= normalization_term[None, :, None]

            expectations += [torch.exp(log_expectations_event)]
            transition_expectations += [torch.exp(log_transition_expectations_event)]
            log_alphas += [log_alphas_event]
            log_betas += [log_betas_event]

        return expectations, transition_expectations, log_alphas, log_betas

    def initialize_e_step(self, input_observations, num_periods, output_observations):
        log_betas_event = torch.zeros((num_periods, self.num_states))
        log_alphas_event = torch.zeros((num_periods, self.num_states))
        # log_psi torch.tensor(channel_size, batch_size, output_size)
        log_psi = torch.squeeze(self.get_log_psi(input_observations)).detach()
        obs_distribution = self.get_distribution(input_observations)
        poisson_log_prob = torch.squeeze(obs_distribution.log_prob(output_observations)).detach()
        log_alphas_event[0, :] = torch.log(self.get_initial_dist()) + poisson_log_prob[:, 0]
        return log_alphas_event, log_betas_event, log_psi, poisson_log_prob

    @staticmethod
    def backward_step(log_betas_event, log_psi, poisson_log_prob, t):
        """
        :param log_betas_event: torch.tensor(batch_size, output_size=num_states)
        :param log_psi: torch.tensor(channel_size=num_states, batch_size, output_size=num_states)
        :param poisson_log_prob: torch.tensor(channel_size=num_states, batch_size, output_size=1)
        :param t: int num_periods - t
        :return:
        """
        log_beta_term = log_psi[:, t, :] + log_betas_event[t, :][None, :] + poisson_log_prob[:, t][None, :]
        log_sum1 = torch.logsumexp(log_beta_term, dim=-1)
        # log_sum1 = torch.tensor(channel_size=num_states, batch_size)
        log_betas_event[t - 1, :] = log_sum1

    @staticmethod
    def forward_step(log_alphas_event, log_psi, poisson_log_prob, t):
        """
        :param log_alphas_event: torch.tensor(batch_size, output_size=num_states)
        :param log_psi: torch.tensor(channel_size=num_states, batch_size, output_size=num_states)
        :param poisson_log_prob: torch.tensor(channel_size=num_states, batch_size, output_size=1)
        :param t: int
        :return:
        """
        log_alpha_term = torch.zeros(log_psi[:, t, :].shape) + log_alphas_event[t - 1, :][:, None]
        log_sum0 = torch.logsumexp(log_psi[:, t, :] + log_alpha_term, dim=0)
        # log_sum0 = torch.tensor(channel_size=num_states, batch_size)
        log_alphas_event[t, :] = torch.squeeze(poisson_log_prob[:, t]) + log_sum0

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
        self.psudo_counts = torch.zeros(self.num_states)
        for p in range(len(event_data)):
            self.psudo_counts += expectations[p].sum(dim=0)
        ll_output, ll_state = None, None
        for i in range(20):
            ll_output, ll_state = 0., 0.
            for p in range(len(event_data)):
                input_observations, output_observations = event_data[p]
                self.output_optimizer.zero_grad()
                loss_output = -self.log_likelihood_emissions(input_observations, output_observations, expectations[p])
                loss_output.backward()
                self.output_optimizer.step()
                ll_output += loss_output.item()

                self.state_optimizer.zero_grad()
                loss_state = -self.log_likelihood_transitions(input_observations, transition_expectations[p])
                loss_state.backward()
                self.state_optimizer.step()
                ll_state += loss_state.item()

        return - ll_state - ll_output

    def fit(self, event_data, num_iterations=200, save=True):
        """Fit an IOHMM using the EM algorithm above.

        Returns
        -------
        lls: the marginal log likelihood over EM iterations
        parameters: the final parameters
        """
        lls = []
        c = 0
        # print("Solving", end="", flush=True)
        print("Solving")
        improvement = 10
        while improvement > -1e-4:
            # E-Step
            expectations, transition_expectations, _, _ = self.e_step(event_data)
            # M-Step
            lls_iter = self.m_step(event_data, expectations, transition_expectations)

            if c % 10 == 0:
                print("[iter %3i] Loss: %10.4f" % (c, lls_iter))
            c += 1
            if len(lls) > 0:
                improvement = lls_iter - lls[-1]
            lls += [lls_iter]

            if c == num_iterations:
                print("Reached max iterations!")
                break

        print("Done")

        if save:
            torch.save({
                'model_state_dict': self.output_network.state_dict(),
                'optimizer_state_dict': self.output_optimizer.state_dict(),
            }, PATH_OUTPUT_NETWORK.format(name=self.__model_name))
            torch.save({
                'psudo_counts': self.psudo_counts,
                'model_state_dict': self.state_network.state_dict(),
                'optimizer_state_dict': self.state_optimizer.state_dict(),
            }, PATH_STATE_NETWORK.format(name=self.__model_name))
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
                input_observations = x_data.loc[city].values.reshape(num_periods, -1)
                output_observations = y_data.loc[city].values.reshape(num_periods, -1)
                event_data.append((np2torch(input_observations), np2torch(output_observations)))
        return event_data

    def viterbi(self, event_data):
        most_likely_states = []
        num_events = len(event_data)
        for p in range(num_events):
            input_observations, output_observations = event_data[p]
            num_periods = input_observations.shape[0]
            log_mu_event = torch.zeros((num_periods, self.num_states))
            # log_psi torch.tensor(channel_size, batch_size, output_size)
            log_psi = torch.squeeze(self.get_log_psi(input_observations)).detach()
            obs_distribution = self.get_distribution(input_observations)
            poisson_log_prob = torch.squeeze(obs_distribution.log_prob(output_observations)).detach()
            for t in range(1, num_periods+1):
                log_mu_term = log_psi[:, num_periods-t, :] + log_mu_event[num_periods-t, :][None, :]
                # log_sum1 = torch.tensor(channel_size=num_states, batch_size)
                log_mu_next = torch.squeeze((poisson_log_prob[:, num_periods-t][None, :] + log_mu_term).max(dim=-1).values)
                log_mu_event[num_periods-t-1, :] = log_mu_next

            most_likely_states_batch = [None] * num_periods
            log_mu_term = poisson_log_prob[:, 0] + log_mu_event[0, :][None, :]
            most_likely_states_batch[0] = (log_mu_term + torch.log(self.get_initial_dist())).argmax().item()
            for t in range(1, num_periods):
                log_mu_term = poisson_log_prob[:, t] + log_mu_event[t, :][None, :]
                prev_state = most_likely_states_batch[t-1]
                log_transition = torch.squeeze(log_psi[prev_state, t, :])
                most_likely_states_batch[t] = (log_mu_term + log_transition).argmax().item()

            most_likely_states += [most_likely_states_batch]

        return most_likely_states

    def predict(self, event_data):
        most_likely_states = self.viterbi(event_data=event_data)
        expectations, transition_expectations, _, _ = self.e_step(event_data)
        output_viterbi = []
        output_network = []
        for p in range(len(event_data)):
            input_observations, output_observations = event_data[p]
            log_rates = self.output_network.forward(input_observations)
            prediction = torch.squeeze(torch.exp(log_rates)).detach().numpy()
            output_network += [prediction]
            output_viterbi += [prediction[(most_likely_states[p], np.arange(output_observations.shape[0]))]]

        return output_viterbi, output_network, most_likely_states, expectations

    def forecast(self, train_event_data, test_event_data, m=8, num_samples=250, alpha=0.05):
        state_space = list(range(self.num_states))
        forecasts = []
        states_prob = []
        for p in range(len(train_event_data)):
            input_obs_train, output_obs_train = train_event_data[p]
            input_obs_test, output_obs_test = test_event_data[p]
            num_periods = input_obs_train.shape[0]
            test_periods = input_obs_test.shape[0]
            forecasts_event = pd.DataFrame(np.nan, index=np.arange(test_periods), columns=['map', 'lower', 'upper'])
            states_prob_event = pd.DataFrame(np.nan, index=np.arange(test_periods), columns=state_space)
            input_observations = input_obs_train.clone()
            output_observations = output_obs_train.clone()

            print("Sampling", end="", flush=True)
            for t in range(input_obs_test.shape[0]-m):
                log_filter_prob = self.filter(input_observations, num_periods,
                                              output_observations)

                log_psi_m = torch.squeeze(self.get_log_psi(input_obs_test[t:t+m, :])).detach()
                log_expectations_event = torch.zeros((m+1, self.num_states))
                log_expectations_event[0, :] = log_filter_prob[-1, :]
                for step in range(m):
                    log_expectations_term = log_expectations_event[step, :][:, None] + log_psi_m[:, step, :]
                    log_expectations_event[step+1, :] = torch.squeeze(torch.logsumexp(log_expectations_term, dim=0))

                state_dist = torch.exp(log_expectations_event[-1, :]).detach().numpy()
                states_prob_event.at[t+m] = state_dist
                states_sim = np.random.choice(state_space, size=num_samples, p=state_dist)
                input_observation_t = torch.reshape(input_obs_test[t, :], (1, -1))
                obs_distribution = self.get_distribution(input_observation_t)
                samples_sim = torch.squeeze(obs_distribution.sample((num_samples,))).detach().numpy()

                obs_sim = samples_sim[(np.arange(num_samples), states_sim)]
                lower_value = np.percentile(obs_sim, q=100*alpha/2)
                upper_value = np.percentile(obs_sim, q=100*(1.-alpha/2))
                map_value = np.median(obs_sim)
                forecasts_event.at[t+m, 'lower'] = lower_value
                forecasts_event.at[t+m, 'upper'] = upper_value
                forecasts_event.at[t+m, 'map'] = map_value

                if t % 10 == 0:
                    print(".", end="", flush=True)
                num_periods += 1
                input_observations = torch.cat((input_observations, input_observation_t), dim=0)
                output_observations = torch.cat((output_observations, torch.reshape(output_obs_test[t], (1, -1))), dim=0)

            print("Done")
            forecasts += [forecasts_event]
            states_prob += [states_prob_event]

        return forecasts, states_prob

    def filter(self, input_observations, num_periods, output_observations):
        initial_eval = self.initialize_e_step(input_observations, num_periods, output_observations)
        log_alphas_event, _, log_psi, poisson_log_prob = initial_eval
        for t in range(1, num_periods):
            self.forward_step(log_alphas_event, log_psi, poisson_log_prob, t)

        log_filter_prob = log_alphas_event - torch.logsumexp(log_alphas_event, dim=1)[:, None]
        return log_filter_prob


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    import matplotlib.pyplot as plt
    os.chdir('../')
    dda = DengueDataApi()
    x_train, x_validate, y_train, y_validate = dda.split_data()
    z_train, z_validate, pct_var = dda.get_pca(x_train, x_validate, num_components=4)
    model = IOHMM(num_states=3, num_features=z_train.shape[1], load=True)

    event_data_train = model.format_event_data(z_train.droplevel('year'), y_train.droplevel('year'))
    event_data_test = model.format_event_data(z_validate.droplevel('year'), y_validate.droplevel('year'))
    lls = model.fit(event_data=event_data_train, save=True)
    # plt.plot(lls)
    # plt.show()

    output_viterbi, output_network, most_likely_states, expectations = model.predict(event_data_train)
    forecasts, states_prob = model.forecast(event_data_train, event_data_test)





