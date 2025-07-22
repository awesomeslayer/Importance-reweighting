import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

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
