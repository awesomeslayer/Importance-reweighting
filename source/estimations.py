import logging
import numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from .KL_LSCV import KL_find_bw
from scipy.stats import gaussian_kde

log = logging.getLogger("__main__")


def density_estimation(conf, hyp_params_dict, test_gen_dict, bw):
    if bw == "KL":
        bw_temp = KL_find_bw(
            conf,
            test_gen_dict["g_train"],
            test_gen_dict["p_test"],
            hyp_params_dict["beta"],
            hyp_params_dict["KL_flag"],
            hyp_params_dict["estim_type"],
        )
    else:
        bw_temp = bw
    log.debug(f"bw = {bw}, bw_temp = {bw_temp}")

    if hyp_params_dict["estim_type"] == "sklearn":
        kde = KernelDensity(kernel="gaussian", bandwidth=bw_temp).fit(
            test_gen_dict["g_train"]
        )
        g_estim = lambda X: kde.score_samples(X)
    elif hyp_params_dict["estim_type"] == "scipy":
        kde = gaussian_kde(test_gen_dict["g_train"].T, bw_method=bw_temp)
        g_estim = lambda X: np.log(kde.evaluate(X.T))

    return g_estim


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


def rmse(x_err, y_err):
    return np.sqrt(np.mean((x_err - y_err) ** 2))


def mape(x_err, y_err):
    return np.mean(100 * np.abs(x_err - y_err) / y_err)
