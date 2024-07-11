import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from sklearn.mixture import GaussianMixture
import GPy

np.seterr(divide="ignore")
np.random.seed(42)


class DummyModel:
    def __init__(self):
        self.func = None

    def fit(self, X, y):
        self.func = random_linear_func(
            {"n_dim": X.shape[1], "max_mu": np.max(X.sum(axis=1))}
        )

    def predict(self, X):
        return self.func(X)


def random_covariance_matrix(n_dim):
    """
    :param n_dim:
    :return: random positive-definite symmetric matrix n_dim X n_dim with eigenvalues >= 1
    """
    var = np.random.random((n_dim, n_dim))
    var = np.dot(var, var.T) + np.identity(n_dim)
    var /= var.max()
    return var


def _compute_precisions_cholesky(covariances):
    """Manually computes attribute for GMM class"""

    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        cov_chol = cholesky(covariance, lower=True)
        precisions_chol[k] = solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T

    return precisions_chol


def _GMM_gen(max_mu, max_cov, n_components, n_dim):
    mu = max_mu * np.random.random((n_components, n_dim))
    var = max_cov * np.array(
        [random_covariance_matrix(n_dim) for _ in range(n_components)], np.float32
    )

    p = np.random.random(n_components)
    p /= p.sum()

    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.weights_ = p
    gmm.means_ = mu
    gmm.covariances_ = var
    gmm.precisions_cholesky_ = _compute_precisions_cholesky(var)

    return gmm


def random_GMM_samples(config):
    """
    :param config:
    :return: random GaussianMixture samples with center in [0; max_mu) X [0; max_mu) and covariances up to max_cov, log-density function
    """
    gmm = _GMM_gen(
        config["max_mu"], config["max_cov"], config["n_components"], config["n_dim"]
    )
    samples = gmm.sample(config["n_samples"])

    log_density = lambda X: gmm.score_samples(X)

    return samples[0], log_density


def random_uniform_samples(config, fixed_region: bool = False):
    """
    :param config:
    :param fixed_region: use all [0; max_mu) X [0; max_mu) square as domain
    :return: sample, log-density function
    """

    a, b = 0, 0
    if fixed_region:
        a = np.zeros(config["n_dim"])
        b = np.ones(config["n_dim"]) * config["max_mu"]
    else:
        a = np.random.uniform(0, config["max_mu"], config["n_dim"])
        b = np.random.uniform(0, config["max_mu"], config["n_dim"])
        a, b = np.where(a < b, a, b), np.where(a < b, b, a)

    w_a = np.random.random((config["n_samples"], config["n_dim"]))
    w_b = 1 - w_a

    log_density = (
        lambda X: np.log(((a < X) & (X < b)).all(axis=1)) - np.log((b - a)).sum()
    )

    return w_a * a + w_b * b, log_density


def random_GP_func(config):
    u_config = config
    u_config["n_samples"] = config["n_components"]

    X = random_uniform_samples(u_config)[0]
    Y = np.random.randn(config["n_components"], 1)

    # print(f"X:{X}")
    # print(f"Y:{Y}")
    kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=1.0)
    model = GPy.models.GPRegression(X, Y, kernel, noise_var=1e-10)
    fun = lambda X: model.posterior_samples_f(X, full_cov=True, size=1)

    # heatplot(fun)
    return fun


def random_linear_func(config):
    """
    :param config:
    :return:
    """
    x = np.random.random(config["n_dim"]) * 2 - 1
    a = np.random.random() * config["max_mu"]
    return lambda X: ((X * x).sum(axis=1) + a)


def random_gaussian_mixture_func(config):
    """
    :param config:
    :return: GMM log-density
    """
    _, g = random_GMM_samples(config)

    return g


def visualize_GMM_config(config, alpha):
    GMM, _ = random_GMM_samples(config)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(GMM[:, 0], GMM[:, 1], alpha=alpha)

    ax.set_xlim((0, config["max_mu"]))
    ax.set_ylim((0, config["max_mu"]))
    plt.show()
