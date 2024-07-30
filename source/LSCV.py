import numpy as np
import statsmodels.api as sm  # for library lscv, beta = 0
import logging
from scipy.integrate import dblquad
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

log = logging.getLogger("__main__")


def u_mean(log_f_n, p_sample):
    return np.sum(log_f_n(p_sample))


def squared_error_sklearn(h, conf, g_sample, p_sample, beta):
    h = h[0]
    kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(g_sample)
    f_n_squared = lambda x, y: (np.exp(kde.score_samples(np.array([[x, y]])))) ** 2
    log_f_n = lambda X: (kde.score_samples(X))

    def f_sub_sample_mean():
        summation = 0
        for i in range(len(g_sample)):
            subsample = np.delete(g_sample, i, axis=0)

            kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(subsample)
            predict_value = np.exp(kde.score_samples(g_sample[i].reshape(1, -1)))
            summation += predict_value
        return summation / len(g_sample)

    integral = dblquad(f_n_squared, 0, conf["max_mu"], 0, conf["max_mu"])
    uniform_sum = u_mean(log_f_n, p_sample)
    sub = f_sub_sample_mean()
    cv = integral[0] - 2 * sub - beta * uniform_sum
    log.debug(
        f"integral = {integral}, subsample = {sub}, uniform_sum = {uniform_sum}, cv = {cv}"
    )
    return cv


def KL_LSCV_find_bw(conf, g_sample, p_sample, beta=0):
    h0 = np.array(g_sample).std() * (len(g_sample) ** (-0.2))

    log.debug(f"Initial h0: {h0}")
    cons = {"type": "ineq", "fun": lambda x: x[0] - 10 ** (-8)}

    h = minimize(
        squared_error_sklearn,
        h0,
        args=(conf, g_sample, p_sample, beta),
        constraints=cons,
    ).x

    return h[0]
