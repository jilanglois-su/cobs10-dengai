import numpy as np
from scipy.optimize import minimize
from src.d04_modeling.abstract_model import AbstractModel
from tqdm import tqdm
from scipy.stats import poisson


class PoissonGLM(AbstractModel):

    def __init__(self, x_train, y_train, sigma2):
        super(PoissonGLM, self).__init__(x_train=x_train, y_train=y_train)
        self.__cov_map = None
        self.__sigma2 = sigma2
        self.__num_rows = len(x_train)

    def get_cov_map(self):
        return self.__cov_map

    def log_joint(self, y, X, w=None, sigma2=None):
        if w is None:
            w = self.__w_map
        if sigma2 is None:
            sigma2 = self.__sigma2
        ljp = -np.dot(w, w)/(2 * sigma2)
        ljp += np.dot(y, np.dot(X, w))
        ljp -= np.sum(np.exp(np.dot(X, w)))

        return ljp

    @staticmethod
    def log_joint_grad(y, X, w, sigma2):
        return -w/sigma2 + np.dot(X.T, y) - np.dot(X.T, np.exp(np.dot(X, w))).T

    @staticmethod
    def log_joint_hess(y, X, w, sigma2):
        return -np.eye(len(w)) / sigma2 - np.dot(X.T, np.multiply(X, np.exp(np.dot(X, w))[:, np.newaxis]))

    def compute_posterior_mode(self):
        # Minimize the log joint. Normalize by N so it's better scaled.
        x_train = self.get_x_train()
        y_train = self.get_y_train()
        obj = lambda w: -self.log_joint(y_train, x_train, w, self.__sigma2) / self.__num_rows
        obj_grad = lambda w: -self.log_joint_grad(y_train, x_train, w, self.__sigma2) / self.__num_rows

        # Keep track of the weights visited during optimization.
        w_init = np.zeros(x_train.shape[1])
        w_hist = [w_init]

        def callback(w):
            w_hist.append(w)

        result = minimize(obj, w_init, jac=obj_grad, callback=callback, method="BFGS")
        w_hist = np.array(w_hist)

        self.__w_map = w_hist[-1]

        return result, w_hist

    def sample_posterior_w(self, num_samples):
        w_s = np.random.multivariate_normal(mean=self.__w_map, cov=self.__cov_map, size=num_samples)
        return w_s

    def compute_laplace_approximation(self):
        y_train = self.get_y_train()
        x_train = self.get_x_train()
        self.__cov_map = -np.linalg.inv(self.log_joint_hess(y_train, x_train, self.__w_map, self.__sigma2))
        return None

    def get_posterior_predictive_distribution(self, x_validate, y_validate, ncols, num_samples):
        max_count = y_validate[:ncols].max()
        num_validate = len(x_validate)
        posterior_predictive_distribution = np.zeros([max_count + 1, num_validate])
        w_s = self.sample_posterior_w(num_samples)
        lambda_n = np.exp(np.dot(x_validate, w_s.T))  # num_validate x num_samples matrix
        for k in tqdm(range(max_count + 1)):
            posterior_predictive_distribution[k, :] = poisson.pmf(k, mu=lambda_n).mean(axis=1).T
        return posterior_predictive_distribution


if __name__ == "__main__":
    import os
    import seaborn as sns
    import pandas as pd
    from src.d01_data.dengue_data_api import DengueDataApi
    os.chdir('../')
    dda = DengueDataApi()
    x_train, x_validate, y_train, y_validate = dda.split_data()
    sigma2 = 1.
    poisson_glm = PoissonGLM(x_train=x_train, y_train=y_train, sigma2=sigma2)

    _, w_hist = poisson_glm.compute_posterior_mode()
    w_hist_df = pd.DataFrame(w_hist, columns=x_train.columns)
    w_hist_df['log_joint'] = w_hist_df.apply(lambda w: poisson_glm.log_joint(y_train, x_train,
                                                                             w.values, sigma2), axis=1)
    w_hist_df.name = 'iter'
    axs1 = sns.lineplot(data=w_hist_df.iloc[1:].reset_index(), x="index", y="log_joint")

    poisson_glm.compute_laplace_approximation()
    cov_map = poisson_glm.get_cov_map()
    cov_map_df = pd.DataFrame(cov_map, index=x_train.columns, columns=x_train.columns)
    axs2 = sns.heatmap(cov_map_df)

    poisson_glm.validate_model(x_validate=x_validate, y_validate=y_validate)
