from abc import ABC
import pandas as pd
import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Construct the model
class StateSpaceModel(sm.tsa.statespace.MLEModel, ABC):
    def __init__(self, endog, factors_x, factors_y):
        # Initialize the state space model
        k_states = k_posdef = factors_x + factors_y
        super(StateSpaceModel, self).__init__(endog, k_states=k_states, k_posdef=k_posdef,
                                              initialization='approximate_diffuse')
        self._covariates = endog.columns
        self._factors_x = factors_x
        self._factors_y = factors_y
        self._param_cov_state_f_idx = self._params_cov_state_z_idx = self._params_cov_obs_idx = None
        self._params_phi_1_idx = self._params_phi_23_idx = self._params_phi_23_idx = None
        self._params_a_1_idx = self._params_a_2_idx = None

        # Setup the fixed components of the state space representation
        transition_f = np.hstack((np.ones((factors_x, factors_x)), np.zeros((factors_x, factors_y))))
        transition_z = np.ones((factors_y,  k_states))
        transition = np.vstack((transition_f, transition_z))
        dims_x = endog.shape[1]-1
        dims_y = 1
        self._dims_x = dims_x
        self._dims_y = dims_y
        # Assume [x, y]'
        design_x = np.hstack((np.ones((dims_x, factors_x)), np.zeros((dims_x, factors_y))))
        design_y = np.hstack((np.zeros((dims_y, factors_x)), np.ones((dims_y, factors_y))))
        design = np.vstack((design_x, design_y))

        self.ssm['design'] = design.reshape((dims_x + 1, k_states, 1))
        self.ssm['transition'] = transition.reshape((k_states, k_states, 1))
        self.ssm['selection'] = np.eye(k_states)
        self.ssm['obs_intercept'] = np.zeros(dims_x + dims_y).reshape(-1, 1)

        # Cache some indices
        self._state_cov_idx = np.diag_indices(k_posdef)
        self._obs_cov_idx = np.diag_indices(dims_x + dims_y)

        grid_transition_f = (np.repeat(np.arange(factors_x), factors_x),
                             np.tile(np.arange(factors_x), factors_x))
        grid_transition_z = (np.repeat(np.arange(factors_x, k_states), k_states),
                             np.tile(np.arange(k_states), factors_y))
        self._transition_f_idx = grid_transition_f
        self._transition_z_idx = grid_transition_z

        grid_design_x = (np.repeat(np.arange(dims_x), factors_x),
                         np.tile(np.arange(factors_x), dims_x))
        grid_design_y = (np.repeat(np.arange(dims_x, dims_x + dims_y), factors_y),
                         np.tile(np.arange(factors_x, k_states), dims_y))
        self._design_x_idx = grid_design_x
        self._design_y_idx = grid_design_y

        self.init_param_indx()

    @staticmethod
    def get_position(idx, i, row_offset=0, col_offset=0):
        return idx[0][i]-row_offset, idx[1][i]-col_offset

    def init_param_indx(self):
        c = 0
        params_cov_obs = ['sigma2.%s' % i for i in self._covariates]
        self._params_cov_obs_idx = (c, c + len(params_cov_obs))
        c += len(params_cov_obs)
        params_cov_state_f = ['sigma2.f.%i' % i for i in range(self._factors_x)]
        self._param_cov_state_f_idx = (c, c + len(params_cov_state_f))
        c += len(params_cov_state_f)
        params_cov_state_z = ['sigma2.z.%i' % i for i in range(self._factors_y)]
        self._params_cov_state_z_idx = (c, c + len(params_cov_state_z))
        c += len(params_cov_state_z)
        params_cov = params_cov_state_f + params_cov_state_z + params_cov_obs

        params_phi_1 = ['phi.1.%i%i' % self.get_position(self._transition_f_idx, i) for i in range(len(self._transition_f_idx[0]))]
        self._params_phi_1_idx = (c, c+len(params_phi_1))
        c += len(params_phi_1)

        params_phi_23 = ['phi.23.%i%i' % self.get_position(self._transition_z_idx, i,
                                                           row_offset=self._factors_x) for i in range(len(self._transition_z_idx[0]))]
        self._params_phi_23_idx = (c, c + len(params_phi_23))
        c += len(params_phi_23)
        params_phi = params_phi_1 + params_phi_23

        params_a_1 = ['a.1.%i%i' % self.get_position(self._design_x_idx, i) for i in range(len(self._design_x_idx[0]))]
        self._params_a_1_idx = (c, c+len(params_a_1))
        c += len(params_a_1)
        params_a_2 = ['a.2.%i%i' % self.get_position(self._design_y_idx, i,
                                                     row_offset=self._dims_x,
                                                     col_offset=self._factors_x) for i in range(len(self._design_y_idx[0]))]
        self._params_a_2_idx = (c, c+len(params_a_2))
        c += len(params_a_2)
        params_a = params_a_1 + params_a_2

        return params_cov + params_phi + params_a

    @property
    def param_names(self):
        return self.init_param_indx()

    # Describe how parameters enter the model
    def update(self, params, *args, **kwargs):
        params = super(StateSpaceModel, self).update(params, *args, **kwargs)
        # Observation covariance
        self.ssm[('obs_cov',) + self._obs_cov_idx] = params[self._params_cov_obs_idx[0]:self._params_cov_obs_idx[1]]
        # State covariance
        self.ssm[('state_cov',) + self._state_cov_idx] = params[self._param_cov_state_f_idx[0]:self._params_cov_state_z_idx[1]]
        # Transition matrix
        self.ssm[('transition',) + self._transition_f_idx] = params[self._params_phi_1_idx[0]:self._params_phi_1_idx[1]]
        self.ssm[('transition',) + self._transition_z_idx] = params[self._params_phi_23_idx[0]:self._params_phi_23_idx[1]]
        # Design matrix
        self.ssm[('design',) + self._design_x_idx] = params[self._params_a_1_idx[0]:self._params_a_1_idx[1]]
        self.ssm[('design',) + self._design_y_idx] = params[self._params_a_2_idx[0]:self._params_a_2_idx[1]]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        design, obs_cov, state_cov, transition = self.generate_start_matrices()

        params_state_cov = state_cov[self._state_cov_idx]
        params_obs_cov = obs_cov[self._obs_cov_idx]

        params_phi = np.concatenate((transition[self._transition_f_idx],
                                     transition[self._transition_z_idx]), axis=0)
        params_a = np.concatenate((design[self._design_x_idx],
                                   design[self._design_y_idx]), axis=0)

        return np.concatenate((params_obs_cov, params_state_cov, params_phi, params_a))

    def generate_start_matrices(self):
        _exog = pd.DataFrame(self.endog[:, :-1], columns=self._covariates[:-1]).interpolate()
        _endog = pd.Series(self.endog[:, -1], name=self._covariates[-1]).interpolate()
        _, _, vh = np.linalg.svd(_exog, full_matrices=True)
        factors = pd.DataFrame(np.dot(_exog, vh[:, :self._factors_x]), index=_exog.index)
        _model = SARIMAX(endog=_endog, exog=factors, order=(self._factors_y, 0, 0))
        res = _model.fit(disp=False, maxiter=100)
        params_arx = res.params

        phi1 = np.eye(self._factors_x)
        factors_coeff = params_arx.values[:self._factors_x].reshape(1, -1)
        ar_coeff = params_arx.values[self._factors_x:-1].reshape(1, -1)
        phi2 = np.vstack([factors_coeff, np.zeros((self._factors_x - 1, self._factors_x))])
        phi3 = np.vstack([ar_coeff, np.eye(self._factors_y)[:-1, :]])
        transition = np.vstack([np.hstack([phi1, np.zeros((self._factors_y, self._factors_y))]),
                                np.hstack([phi2, phi3])])

        a1 = vh[:, :self._factors_x]
        a2 = np.eye(self._dims_y, self._factors_y)
        design_x = np.hstack([a1, np.zeros((self._dims_x, self._factors_y))])
        design_y = np.hstack([np.zeros((self._dims_y, self._factors_x)), a2])
        design = np.vstack([design_x, design_y])

        state_cov = np.eye(self.k_states)

        obs_cov = np.eye(len(self._covariates))
        obs_cov[-1, -1] = params_arx.values[-1]

        return design, obs_cov, state_cov, transition

    def transform_params(self, unconstrained):
        constrained = unconstrained
        for i1, i2 in [self._param_cov_state_f_idx, self._params_cov_state_z_idx, self._params_cov_obs_idx]:
            constrained[i1:i2] = unconstrained[i1:i2] ** 2
        return constrained

    def untransform_params(self, constrained):
        unconstrained = constrained
        for i1, i2 in [self._param_cov_state_f_idx, self._params_cov_state_z_idx, self._params_cov_obs_idx]:
            unconstrained[i1:i2] = constrained[i1:i2] ** 0.5
        return unconstrained


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    from src.d04_modeling.abstract_sm import AbstractSM
    os.chdir('../')
    dda = DengueDataApi(interpolate=False)
    x1, x2, y1, y2 = dda.split_data(random=False)

    abstract_model = AbstractSM(x_train=x1, y_train=y1, bias=False)
    for city in abstract_model.get_cities():
        endog, exog = abstract_model.format_data_arimax(x1.loc[city], y1.loc[city], interpolate=False)
        endog = pd.concat([exog, endog.to_frame()], axis=1)
        model = StateSpaceModel(endog=endog, factors_x=3, factors_y=3)
        results = model.fit(maxiter=100)
        print(results.params)

