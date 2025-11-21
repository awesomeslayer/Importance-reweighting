import os
from copy import copy

import GPy
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import pdist, squareform
from scipy.stats import poisson
from sklearn.mixture import GaussianMixture
import os
from scipy.stats import poisson
from scipy.spatial.distance import pdist, squareform

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


def _GMM_gen(max_mu, max_cov, n_components, n_dim, mu_offset=None, cov_scale=None):
    """
    Enhanced GMM generator with optional mean offset and covariance scaling
    
    :param max_mu: maximum mean value
    :param max_cov: maximum covariance scale
    :param n_components: number of mixture components
    :param n_dim: dimensionality
    :param mu_offset: optional offset to add to means (array of shape (n_dim,))
    :param cov_scale: optional scaling factor for covariances (scalar)
    :return: GaussianMixture object
    """
    mu = max_mu * np.random.random((n_components, n_dim))
    
    # Apply mean offset if provided
    if mu_offset is not None:
        mu = mu + np.array(mu_offset).reshape(1, -1)
    
    var = max_cov * np.array(
        [random_covariance_matrix(n_dim) for _ in range(n_components)], np.float32
    )
    
    # Apply covariance scaling if provided
    if cov_scale is not None:
        var = var * cov_scale

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
        config["max_mu"], config["max_cov"], config["n_GM_components"], config["n_dim"]
    )
    samples = gmm.sample(config["n_samples"])

    log_density = lambda X: gmm.score_samples(X)

    return samples[0], log_density


def random_GMM_samples_with_params(config, mu_offset=None, cov_scale=None, 
                                   fixed_means=False, fixed_cov=False):
    """
    Generate GMM samples with flexible mean and covariance control
    
    :param config: configuration dictionary
    :param mu_offset: offset to add to means (None, scalar, or array)
    :param cov_scale: scaling factor for covariances (None or scalar)
    :param fixed_means: if True, use zero means
    :param fixed_cov: if True, use identity covariances
    :return: samples, log_density function
    """
    if fixed_means:
        mu = np.full((config["n_GM_components"], config["n_dim"]), 50)

    else:
        mu = config["max_mu"] * np.random.random((config["n_GM_components"], config["n_dim"]))
    
    if mu_offset is not None:
        if np.isscalar(mu_offset):
            mu = mu + mu_offset
        else:
            mu = mu + np.array(mu_offset).reshape(1, -1)
    
    if fixed_cov:
        var = np.array([np.identity(config["n_dim"]) for _ in range(config["n_GM_components"])], np.float32)
    else:
        var = config["max_cov"] * np.array(
            [random_covariance_matrix(config["n_dim"]) for _ in range(config["n_GM_components"])], 
            np.float32
        )
    
    if cov_scale is not None:
        var = var * cov_scale

    p = np.random.random(config["n_GM_components"])
    p /= p.sum()

    gmm = GaussianMixture(n_components=config["n_GM_components"], covariance_type="full")
    gmm.weights_ = p
    gmm.means_ = mu
    gmm.covariances_ = var
    gmm.precisions_cholesky_ = _compute_precisions_cholesky(var)
    
    samples = gmm.sample(config["n_samples"])
    log_density = lambda X: gmm.score_samples(X)

    return samples[0], log_density


def random_GMM_samples_cropped(config, crop_region=None):
    """
    Generate GMM samples and crop to a specific region
    
    :param config: configuration dictionary
    :param crop_region: dict with 'x_min', 'x_max', 'y_min', 'y_max', etc.
                       or string like 'lower_half', 'upper_half', 'left_half', 'right_half'
    :return: cropped samples, log_density function
    """
    gmm = _GMM_gen(
        config["max_mu"], config["max_cov"], config["n_GM_components"], config["n_dim"]
    )
    
    # Generate more samples than needed for cropping
    oversample_factor = 3
    samples_temp = gmm.sample(config["n_samples"] * oversample_factor)[0]
    
    # Define crop region
    if isinstance(crop_region, str):
        if crop_region == 'lower_half':
            mask = samples_temp[:, 1] < config["max_mu"] / 2
        elif crop_region == 'upper_half':
            mask = samples_temp[:, 1] >= config["max_mu"] / 2
        elif crop_region == 'left_half':
            mask = samples_temp[:, 0] < config["max_mu"] / 2
        elif crop_region == 'right_half':
            mask = samples_temp[:, 0] >= config["max_mu"] / 2
        elif crop_region == 'center_square':
            center = config["max_mu"] / 2
            quarter = config["max_mu"] / 4
            mask = ((samples_temp[:, 0] >= center - quarter) & 
                   (samples_temp[:, 0] <= center + quarter) &
                   (samples_temp[:, 1] >= center - quarter) & 
                   (samples_temp[:, 1] <= center + quarter))
        else:
            mask = np.ones(len(samples_temp), dtype=bool)
    elif isinstance(crop_region, dict):
        mask = np.ones(len(samples_temp), dtype=bool)
        for dim in range(config["n_dim"]):
            if f'dim{dim}_min' in crop_region:
                mask &= samples_temp[:, dim] >= crop_region[f'dim{dim}_min']
            if f'dim{dim}_max' in crop_region:
                mask &= samples_temp[:, dim] <= crop_region[f'dim{dim}_max']
    else:
        mask = np.ones(len(samples_temp), dtype=bool)
    
    cropped_samples = samples_temp[mask]
    
    # Ensure we have enough samples
    while len(cropped_samples) < config["n_samples"]:
        additional_samples = gmm.sample(config["n_samples"])[0]
        if isinstance(crop_region, str):
            if crop_region == 'lower_half':
                mask = additional_samples[:, 1] < config["max_mu"] / 2
            elif crop_region == 'upper_half':
                mask = additional_samples[:, 1] >= config["max_mu"] / 2
            elif crop_region == 'left_half':
                mask = additional_samples[:, 0] < config["max_mu"] / 2
            elif crop_region == 'right_half':
                mask = additional_samples[:, 0] >= config["max_mu"] / 2
            else:
                mask = np.ones(len(additional_samples), dtype=bool)
        else:
            mask = np.ones(len(additional_samples), dtype=bool)
        cropped_samples = np.vstack([cropped_samples, additional_samples[mask]])
    
    samples = cropped_samples[:config["n_samples"]]
    log_density = lambda X: gmm.score_samples(X)

    return samples, log_density


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
    u_config = copy(config)
    u_config["n_samples"] = config["n_GP_components"]

    X = random_uniform_samples(u_config)[0]
    Y = np.random.randn(u_config["n_GP_components"], 1)

    kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=1.0)
    model = GPy.models.GPRegression(X, Y, kernel, noise_var=1e-10)

    return lambda X: model.posterior_samples_f(X, full_cov=True, size=1)


def heatmap(config, fun, n_points=50):
    x_range = np.linspace(0, config["max_mu"], 50)
    y_range = np.linspace(0, config["max_mu"], 50)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    predicted_values = fun(grid_points).reshape(50, 50)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, predicted_values, levels=20, cmap="viridis")
    plt.colorbar(label="Prediction")
    plt.title("Gaussian Process Prediction")
    plt.savefig("./plots/results/fun_GP.pdf")
    plt.xlabel("X")
    plt.ylabel("Y")


def random_linear_func(config):
    """
    :param config:
    :return:
    """
    x = np.random.random(config["n_dim"]) * 2 - 1
    a = np.random.random() * config["max_mu"]
    return lambda X: ((X * x).sum(axis=1) + a)


def random_quadratic_func(config):
    """
    Generates a random quadratic function in 2D.

    :param config: Dictionary containing configuration parameters:
                   - "n_dim": Number of dimensions (must be 2 for 2D).
                   - "max_mu": Maximum value for the constant term.
                   - "max_coef": Maximum value for quadratic coefficients.
    :return: A function that takes a 2D input and returns a quadratic form.
    """
    if config["n_dim"] != 2:
        raise ValueError("This function is designed for 2D inputs only.")

    a = np.random.random() * config["max_coef"]  # x^2
    b = np.random.random() * config["max_coef"]  # y^2
    c = np.random.random() * config["max_coef"]  # xy

    d = np.random.random() * config["max_coef"]  # x
    e = np.random.random() * config["max_coef"]  # y

    f = np.random.random() * config["max_mu"]  # Constant

    return lambda X: (
        a * X[0] ** 2  # x^2 term
        + b * X[1] ** 2  # y^2 term
        + c * X[0] * X[1]  # xy term
        + d * X[0]  # x term
        + e * X[1]  # y term
        + f  # constant term
    )


def random_gaussian_mixture_func(config):
    """
    :param config:
    :return: GMM log-density
    """
    _, g = random_GMM_samples(config)

    return g


def visualize_pattern(samples, config, name, alpha=0.7):
    os.makedirs(f"./main/results/patterns/{name}", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha)
    ax.set_title(f"{name}")
    ax.set_xlim((0, config["max_mu"]))
    ax.set_ylim((0, config["max_mu"]))
    plt.savefig(f"./main/results/patterns/{name}/{config['max_cov']}.pdf")


def random_thomas_samples(config):
    """
    Generate Thomas cluster process samples
    
    :param config: dictionary with parameters
        n_samples: exact number of points to generate
        max_mu: window size
        n_dim: dimension of the space (typically 2)
        kappa: parent intensity (default 5/(24*1000))
        mu_offspring: mean offspring per cluster (default 120)
        sigma: standard deviation of offspring displacement (default 7.5)
    :return: samples array, log-density function
    """
    kappa = config.get('kappa', 5/(24*1000))
    mu_offspring = config.get('max_cov', 120)
    sigma = config.get('sigma', 7.5)
    
    points = []
    while len(points) < config['n_samples']:
        n_parents = max(1, poisson.rvs(kappa * config['max_mu']**2))
        parent_points = np.random.uniform(0, config['max_mu'], 
                                        (n_parents, config['n_dim']))
        
        for parent in parent_points:
            if len(points) >= config['n_samples']:
                break
                
            n_offspring = poisson.rvs(mu_offspring)
            
            if n_offspring > 0:
                displacement = np.random.normal(0, sigma, (n_offspring, config['n_dim']))
                offspring = parent + displacement
                
                for i in range(config['n_dim']):
                    offspring[:, i] = np.where(offspring[:, i] < 0, 
                                             -offspring[:, i], 
                                             offspring[:, i])
                    offspring[:, i] = np.where(offspring[:, i] > config['max_mu'],
                                             2*config['max_mu'] - offspring[:, i],
                                             offspring[:, i])        
                points.extend(offspring)
    
    points = np.array(points)[:config['n_samples']]
    
    def log_density(X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        density = np.zeros(len(X))
        for parent in parent_points:
            diff = X - parent
            dist_sq = np.sum(diff**2, axis=1)
            density += np.exp(-dist_sq / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        
        return np.log(kappa * mu_offspring * density)
    
    return points, log_density


def random_matern_samples(config):
    """
    Generate Matern cluster process samples with log-density function.
    
    :param config: dictionary with parameters
        n_samples: number of points to generate
        x_min: minimum x-value of the simulation window
        x_max: maximum x-value of the simulation window
        y_min: minimum y-value of the simulation window
        lambda_parent: density of parent Poisson point process
        lambda_daughter: mean number of points in each cluster
        radius_cluster: radius of cluster disks for daughter points
    :return: samples array (x, y coordinates of points), log-density function
    """
    
    x_min = 0
    x_max = config['max_mu']
    y_min = 0
    y_max = config['max_mu']
    
    lambda_parent = config['lambda_parent']
    lambda_daughter = config['lambda_daughter']
    radius_cluster = config['max_cov']
    
    r_ext = radius_cluster
    x_min_ext = x_min - r_ext
    x_max_ext = x_max + r_ext
    y_min_ext = y_min - r_ext
    y_max_ext = y_max + r_ext
    
    x_delta_ext = x_max_ext - x_min_ext
    y_delta_ext = y_max_ext - y_min_ext
    area_total_ext = x_delta_ext * y_delta_ext
    
    points = []
    
    while len(points) < config['n_samples']:
        numb_points_parent = np.random.poisson(area_total_ext * lambda_parent)
        
        xx_parent = x_min_ext + x_delta_ext * np.random.uniform(0, 1, numb_points_parent)
        yy_parent = y_min_ext + y_delta_ext * np.random.uniform(0, 1, numb_points_parent)
        
        numb_points_daughter = np.random.poisson(lambda_daughter, numb_points_parent)
        numb_points = sum(numb_points_daughter)  # total number of points
        
        theta = 2 * np.pi * np.random.uniform(0, 1, numb_points)  # angular coordinates
        rho = radius_cluster * np.sqrt(np.random.uniform(0, 1, numb_points))  # radial coordinates
        
        xx0 = rho * np.cos(theta)
        yy0 = rho * np.sin(theta)
        
        xx = np.repeat(xx_parent, numb_points_daughter)
        yy = np.repeat(yy_parent, numb_points_daughter)
        
        xx = xx + xx0
        yy = yy + yy0
        
        boole_inside = ((xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max))
        
        xx = xx[boole_inside]
        yy = yy[boole_inside]
        
        new_points = np.vstack((xx, yy)).T
        points.extend(new_points)
    
    points = np.array(points)[:config['n_samples']]
    
    def log_density(X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        density = np.zeros(len(X))
        for parent_x, parent_y in zip(xx_parent, yy_parent):
            diff_x = X[:, 0] - parent_x
            diff_y = X[:, 1] - parent_y
            dist_sq = diff_x**2 + diff_y**2
            density += np.exp(-dist_sq / (2 * radius_cluster**2)) / (2 * np.pi * radius_cluster**2)
        
        return np.log(lambda_parent * lambda_daughter * density)
    return points, log_density