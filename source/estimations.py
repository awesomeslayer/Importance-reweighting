import logging
import numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from .KL_LSCV import KL_find_bw
from scipy.stats import gaussian_kde
import scipy.stats as stats

import math
from cvxopt import matrix, solvers

log = logging.getLogger("__main__")


def density_estimation(conf, hyp_params_dict, test_gen_dict, bw):
    if bw == "KL":
        bw_temp = KL_find_bw(
            conf,
            test_gen_dict["g_train"],
            test_gen_dict["p_train"],
            hyp_params_dict["beta"],
            hyp_params_dict["KL_enable"],
            hyp_params_dict["estim_type"],
        )
    else:
        bw_temp = bw

    if hyp_params_dict["estim_type"] == "sklearn":
        kde = KernelDensity(kernel="gaussian", bandwidth=bw_temp).fit(
            test_gen_dict["g_train"]
        )
        g_estim = lambda X: kde.score_samples(X)
        bw_temp = kde.bandwidth_

    elif hyp_params_dict["estim_type"] == "scipy":
        kde = gaussian_kde(test_gen_dict["g_train"].T, bw_method=bw_temp)
        g_estim = lambda X: np.log(kde.evaluate(X.T))
        bw_temp = kde.covariance_factor()
    
    log.debug(f"bw = {bw}, bw_temp = {bw_temp}")
    return g_estim, bw_temp


def ISE(err, p, g, g_sample):
    return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )


def ISE_deg(err, p, g, g_sample, lam):
    if lam != 0:
        return logsumexp(err(g_sample) + lam * (p(g_sample) - g(g_sample))) - np.log(
            g_sample.shape[0]
        )
    else:
        return logsumexp(err(g_sample)) - np.log(g_sample.shape[0])


def MCE(err, p_sample):
    return logsumexp(err(p_sample)) - np.log(p_sample.shape[0])


def clip(a, b_min, b_max):
    if a < b_min:
        return b_min, 1
    elif a > b_max:
        return b_max, 1
    else:
        return a, 0


def smooth_clip(x, eps):
    return (1 + eps) / (1 + (2 * eps / (1 - eps)) * np.exp(-x))


def ISE_clip(err, p, g, g_sample, eps, smooth_flag=True, thrhold=0.95, clip_step=0.001):
    log = logging.getLogger("__main__")
    if eps == "quantile":
        for eps_temp in np.arange(1 - clip_step, -clip_step, -clip_step):
            clipped_array = []
            num_clipped = 0
            for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
                weight = np.exp(p_elem - g_elem)
                clipped_weight, i = clip(weight, 1 - eps_temp, 1 + eps_temp)
                clipped_array.append(np.log(clipped_weight))
                num_clipped = num_clipped + i

            if num_clipped > len(g_sample) * thrhold:
                eps = eps_temp
                log.debug(f"eps_clip_quantile = {eps}")
                break
    else:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            weight = np.exp(p_elem - g_elem)
            clipped_weight, i = clip(weight, 1 - eps, 1 + eps)
            clipped_array.append(np.log(clipped_weight))

    if smooth_flag:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            weight = np.exp(p_elem - g_elem)
            clipped_array.append(np.log(smooth_clip(weight, eps)))

    return logsumexp(clipped_array + err(g_sample)) - np.log(g_sample.shape[0])


def rmse(x_err, y_err, confidence=0.95):
    errors = [(x - y) ** 2 for x, y in zip(x_err, y_err)]
    rmse_value = np.sqrt(np.mean(errors))

    n = len(errors)
    std_dev = np.sqrt(np.var(errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    return rmse_value, (rmse_value - margin_of_error, rmse_value + margin_of_error)


def mape(x_err, y_err, confidence=0.95):
    errors = [abs(x - y) / y for x, y in zip(x_err, y_err) if y != 0]
    
    mape_value = min(np.mean(errors) * 100, 200)
    if np.isnan(mape_value):
        mape_value = 200

    n = len(errors)
    std_dev = np.sqrt(np.var(errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    return mape_value, (
        mape_value - margin_of_error * 100,
        mape_value + margin_of_error * 100
    )

def kernel_mean_matching(g_train, g_test, kern='lin', B=1.0, eps=None):
    nx = g_train.shape[0]
    nz = g_test.shape[0]
    
    if eps is None:
        eps = max(1e-6, B / np.sqrt(nz))  # Avoid very small values for uniform distributions
    
    if kern == 'lin':
        K = np.dot(g_test, g_test.T)
        kappa = np.sum(np.dot(g_test, g_train.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(g_test, g_test, sigma=adjust_sigma(g_test))
        kappa = np.sum(compute_rbf(g_test, g_train, sigma=adjust_sigma(g_test)), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('Unknown kernel')
    
    K = matrix(K)
    kappa = matrix(kappa)
    
    # Regularization with dynamic epsilon
    G = matrix(np.vstack([np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)]))
    h = matrix(np.hstack([nz * (1 + eps), nz * (eps - 1), B * np.ones(nz), np.zeros(nz)]))
    
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x']).flatten()
    
    # Clip the coefficients to avoid extreme values
    coef = np.clip(coef, 0, B)
    
    return coef

def compute_rbf(X, Z, sigma=1.0):
    """ Compute RBF kernel matrix """
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K

def adjust_sigma(data):
    """ Dynamically adjust sigma based on the variance of the data """
    pairwise_dists = np.sum((data[:, None] - data[None, :])**2, axis=-1)
    median_dist = np.median(pairwise_dists)
    return median_dist / np.log(len(data))  # Use median distance scaled by the log of the sample size

def KMM_error(err, p_sample, g_sample, hyperparam):
    coef = kernel_mean_matching(p_sample, g_sample, kern='rbf', B=hyperparam)
    return logsumexp(err(g_sample) + np.log(coef)) - np.log(g_sample.shape[0])
