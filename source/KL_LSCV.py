import numpy as np
import matplotlib.pyplot as plt

# import statsmodels.api as sm  # for library lscv, beta = 0
import logging
from scipy.integrate import dblquad
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

log = logging.getLogger("__main__")


def u_mean(log_f_n, p_sample):
    return np.mean(log_f_n(p_sample))


def squared_error(h, conf, g_sample, p_sample, beta, flag, estim_type):
    h = h[0]
    if estim_type == "sklearn":
        kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(g_sample)
        log_f_n = lambda X: (kde.score_samples(X))
    elif estim_type == "scipy":
        kde = gaussian_kde(g_sample.T, bw_method=h)
        log_f_n = lambda X: np.log(kde.evaluate(X.T))

    uniform_sum = beta * u_mean(log_f_n, p_sample)
    if flag:
        cv = -uniform_sum
    else:
        if estim_type == "sklearn":
            f_n_squared = lambda x, y: (np.exp(kde.score_samples(np.array([[x, y]])))) ** 2
        elif estim_type == "scipy":
            f_n_squared = lambda x, y: (kde.evaluate(np.array([[x, y]]))) ** 2

        def f_sub_sample_mean():
            summation = 0
            for i in range(len(g_sample)):
                subsample = np.delete(g_sample, i, axis=0)
                if estim_type == "sklearn":
                    sub_kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(subsample)
                    predict_value = np.exp(sub_kde.score_samples(g_sample[i].reshape(1, -1)))
                elif estim_type == "scipy":
                    sub_kde = gaussian_kde(subsample.T, bw_method=h)
                    predict_value = sub_kde.evaluate(g_sample[i].reshape(1, -1))[0]
                summation += predict_value
            return summation / len(g_sample)

        integral = dblquad(f_n_squared, 0, conf["max_mu"], 0, conf["max_mu"])
        sub = f_sub_sample_mean()
        cv = integral[0] - 2 * sub - uniform_sum
        log.debug(f"integral = {integral[0]}, sub = {sub}, uniform_sum = {uniform_sum}")
    log.debug(f"for h = {h}: cv = {cv}")
    return cv


def KL_find_bw(conf, g_sample, p_sample, beta=0, flag=True, estim_type="sklearn"):
    # h_list = np.linspace(0.01, 5, 100)
    
    # best_h = None
    # min_error = float('inf')
    # eps = 1e10
    # log.info(f"starting KL find")
    # for h in h_list:
    #     error = squared_error(h, conf, g_sample, p_sample, beta, flag, estim_type)
    #     if np.isnan(error) or np.isinf(error):
    #         error = eps
    #     log.info(f"cv = {error}")
        
    #     if error < min_error:
    #         min_error = error
    #         best_h = h
    
    # log.debug(f"final h = {best_h}")
    # return best_h
    h0 = np.array(g_sample).std() * (len(g_sample) ** (-0.2))
    log.debug(f"init h = {h0}")
    cons = {"type": "ineq", "fun": lambda x: x[0] - 10 ** (-8)}
    h = minimize(
        squared_error,
        h0,
        args=(conf, g_sample, p_sample, beta, flag, estim_type),
        constraints=cons,
    ).x
    log.debug(f"final h = {h[0]}")
    return h[0]
