import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        m.bias.data.fill_(0.)


class StateNetwork(nn.Module):
    """
    Class for implementing state network
    """
    def __init__(self, in_features, num_states, learning_rate=0.001):
        super().__init__()
        self.lr = learning_rate
        self.network = nn.Linear(in_features=in_features, out_features=num_states)
        self.network.to(device)

    def forward(self, observations):
        output = self.network(observations)

        return output


class OutputNetwork(nn.Module):
    """
    Class for implementing Inference Network
    """
    def __init__(self, in_features, out_features, learning_rate=0.001):
        super().__init__()
        self.lr = learning_rate

        self.network = nn.Linear(in_features=in_features, out_features=out_features)
        self.network.to(device)

    def forward(self, observations):
        output = self.network(observations)

        return output


class Likelihood(nn.Module):
    def __init__(self, num_features, num_states, learning_rate=0.001):
        nn.Module.__init__(self)
        self.num_states = num_states
        self.lr = learning_rate
        self.state_network = OrderedDict()
        self.output_network = OrderedDict()
        for k in range(num_states):
            self.state_network[k] = StateNetwork(in_features=num_features, num_states=num_states)
            self.output_network[k] = OutputNetwork(in_features=num_features, out_features=1)
        self.psudo_counts = nn.Parameter(torch.ones(num_states).to(device))

    def get_initial_dist(self):
        initial_dist = self.psudo_counts / self.psudo_counts.sum()
        return initial_dist

    def get_distribution(self, input_observations):
        log_rates = torch.zeros((input_observations.shape[0], input_observations.shape[1], self.num_states))
        for j in range(self.num_states):
            log_rates_j = self.output_network[j].forward(input_observations)
            log_rates[:, :, [j]] = log_rates_j
        distribution = torch.distributions.Poisson(rate=torch.exp(log_rates))
        return distribution

    def get_log_likelihood(self, event_data, transition_expectations, expectations):
        num_events = len(event_data)
        lls = 0
        for p in range(num_events):
            input_observations, output_observations = event_data[p]
            obs_distribution = self.get_distribution(input_observations)
            poisson_log_prob = torch.squeeze(obs_distribution.log_prob(output_observations))
            expectations_event = expectations[p]
            event_lls_1 = expectations_event * poisson_log_prob
            log_psi = torch.squeeze(self.get_log_psi(input_observations))
            transition_expectations_event = transition_expectations[p]
            event_lls_2 = (transition_expectations_event * log_psi).sum(dim=1)
            lls_event = (event_lls_1 + event_lls_2).sum().sum()
            lls += lls_event

        return lls

    def get_log_psi(self, input_observations):
        intermediate_variables = torch.zeros((input_observations.shape[0], input_observations.shape[1],
                                              self.num_states, self.num_states))
        # transition_matrix as defined by STATS 271 lecture: P_{ji}=P(z_t=i \mid z_{t-1}=j)
        for j in range(self.num_states):
            intermediate_variables[:, :, j, :] = self.state_network[j].forward(input_observations)
        log_psi = F.log_softmax(intermediate_variables, dim=-1)
        return log_psi

