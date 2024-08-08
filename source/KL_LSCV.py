import numpy as np

# import statsmodels.api as sm  # for library lscv, beta = 0
import logging
from scipy.integrate import dblquad
from scipy.optimize import minimize, basinhopping
from sklearn.neighbors import KernelDensity


def u_mean(log_f_n, p_sample):
    return np.mean(log_f_n(p_sample))


def squared_error_sklearn(h, conf, g_sample, p_sample, beta, flag):
    h = h[0]
    kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(g_sample)
    log_f_n = lambda X: (kde.score_samples(X))

    uniform_sum = beta * u_mean(log_f_n, p_sample)
    if flag:
        cv = -uniform_sum
    else:
        f_n_squared = lambda x, y: (np.exp(kde.score_samples(np.array([[x, y]])))) ** 2

        def f_sub_sample_mean():
            summation = 0
            for i in range(len(g_sample)):
                subsample = np.delete(g_sample, i, axis=0)

                kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(subsample)
                predict_value = np.exp(kde.score_samples(g_sample[i].reshape(1, -1)))
                summation += predict_value
            return summation / len(g_sample)

        integral = dblquad(f_n_squared, 0, conf["max_mu"], 0, conf["max_mu"])
        sub = f_sub_sample_mean()
        cv = integral[0] - 2 * sub - uniform_sum

    return cv


def KL_find_bw(conf, g_sample, p_sample, beta=0, flag=True):
    h0 = np.array(g_sample).std() * (len(g_sample) ** (-0.2))
    cons = {"type": "ineq", "fun": lambda x: x[0] - 10 ** (-8)}

    h = minimize(
        squared_error_sklearn,
        h0,
        args=(conf, g_sample, p_sample, beta, flag),
        constraints=cons,
    ).x

    return h[0]
