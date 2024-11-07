import logging
import numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from .KL_LSCV import KL_find_bw
from scipy.stats import gaussian_kde
import scipy.stats as stats

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
