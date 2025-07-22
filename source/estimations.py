import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from cvxopt import matrix, solvers
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
            np.vstack([test_gen_dict["g_train"], test_gen_dict["g_test"]]),
            np.vstack([test_gen_dict["p_train"], test_gen_dict["p_test"]]),
            hyp_params_dict["beta"],
            hyp_params_dict["KL_enable"],
            hyp_params_dict["estim_type"],
        )
    else:
        bw_temp = bw

    if hyp_params_dict["estim_type"] == "sklearn":
        kde = KernelDensity(kernel="gaussian", bandwidth=bw_temp).fit(
            np.vstack([test_gen_dict["g_train"], test_gen_dict["g_test"]])
        )
        g_estim = lambda X: kde.score_samples(X)
        bw_temp = kde.bandwidth_

    elif hyp_params_dict["estim_type"] == "scipy":
        kde = gaussian_kde(
            np.vstack([test_gen_dict["g_train"], test_gen_dict["g_test"]]).T,
            bw_method=bw_temp,
        )
        g_estim = lambda X: np.log(kde.evaluate(X.T))
        bw_temp = kde.covariance_factor()
    
    log.debug(f"bw = {bw}, bw_temp = {bw_temp}")
    return g_estim, bw_temp


def ISE(err, p, g, g_sample):
    return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )


def Classifier_error(g_train, p_train, g_test, p_test, err_func):
    epsilon_prob = 0  
    epsilon_denom = 1e-10  

    if len(g_train) == 0:
        print(
            "Warning: g_train is empty. Cannot compute Classifier_error reliably. Returning NaN."
        )
        return np.nan

    if len(p_train) == 0:
        print(
            "Warning: p_train is empty. Density ratio weights are ill-defined. Returning NaN."
        )
        return np.nan

    X_train = np.vstack([g_train, p_train])
    y_train = np.concatenate([np.ones(len(g_train)), np.zeros(len(p_train))])

    clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3
    )  
    clf.fit(X_train, y_train)

    class_1_idx = -1
    try:
        class_1_idx = np.where(clf.classes_ == 1)[0][0]
    except IndexError:
       
        raise ValueError(
            "Class 1 (source) or Class 0 (target) not found in domain classifier's classes."
        )

    probs_s_eq_1_for_g_test = clf.predict_proba(g_test)[:, class_1_idx]

    N_g_train = float(len(g_train))
    N_p_train = float(len(p_train))

    prior_ratio_mult = N_g_train / N_p_train

    p_s1_clipped = probs_s_eq_1_for_g_test
    p_s0_clipped = 1.0 - p_s1_clipped  

    likelihood_ratio_from_clf = p_s0_clipped / p_s1_clipped

    raw_weights = prior_ratio_mult * likelihood_ratio_from_clf

    original_weights = raw_weights.copy()  
    weights = np.clip(raw_weights, 1e-7, 1000)  

    clipped_count = np.sum(original_weights != weights)
    log_weights = np.log(weights)

    errors = err_func(g_test)
    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)

    if len(errors) != len(g_test):
        raise ValueError(
            f"Length of errors ({len(errors)}) from err_func does not match length of g_test ({len(g_test)})."
        )

    if (
        len(g_test) == 0
    ):  
        print(
            "Warning: g_test is empty. Cannot compute final weighted error. Returning NaN."
        )
        return np.nan

    terms_for_lse = errors + log_weights

    weighted_error = logsumexp(terms_for_lse) - np.log(float(len(g_test)))

    return weighted_error


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


import numpy as np
from scipy import stats
from scipy.stats import spearmanr


def rmse(x_err, y_err, confidence=0.95):
    errors = [(x - y) ** 2 for x, y in zip(x_err, y_err)]
    rmse_value = np.sqrt(np.mean(errors))

    n = len(errors)
    std_dev = np.sqrt(np.var(errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    return rmse_value, (rmse_value - margin_of_error, rmse_value + margin_of_error)


def rmspe(x_err, y_err, confidence=0.95):
    relative_errors = []
    for x, y in zip(x_err, y_err):
        if y == 0:
            continue  
        relative_error = ((x - y) / y) ** 2
        relative_errors.append(relative_error)

    if not relative_errors:  
        return None, (None, None)

    rmspe_value = np.sqrt(np.mean(relative_errors)) * 100

    n = len(relative_errors)
    std_dev = np.sqrt(np.var(relative_errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n)) * 100

    return rmspe_value, (rmspe_value - margin_of_error, rmspe_value + margin_of_error)


def corr(x_err, y_err, confidence=0.95, method="spearman"):
    x_clean, y_clean = [], []
    for x, y in zip(x_err, y_err):
        if not np.isnan(x) and not np.isnan(y):
            x_clean.append(x)
            y_clean.append(y)

    if len(x_clean) < 2:
        return 0.0, (0.0, 0.0) 

    corr_value, p_value = spearmanr(x_clean, y_clean)

    n = len(x_clean)
    if method == "pearson":
        
        z = np.arctanh(corr_value)
        se = 1 / np.sqrt(n - 3)
    else:
        z = np.arctanh(corr_value)
        se = 1.06 / np.sqrt(n - 3)  

    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha / 2)
    lower_z = z - z_critical * se
    upper_z = z + z_critical * se

    lower = np.tanh(lower_z)
    upper = np.tanh(upper_z)

    return corr_value, (lower, upper)


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
        eps = max(1e-6, B / np.sqrt(nz))  
    
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
    
    G = matrix(np.vstack([np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)]))
    h = matrix(np.hstack([nz * (1 + eps), nz * (eps - 1), B * np.ones(nz), np.zeros(nz)]))
    
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x']).flatten()
    
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
