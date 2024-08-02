import logging
import numpy as np
from scipy.special import logsumexp

def importance_sampling_error(err, p, g, g_sample):
    return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )


def importance_sampling_error_degree(err, p, g, g_sample, lam):
    if lam != 0:
        return logsumexp(err(g_sample) + lam * (p(g_sample) - g(g_sample))) - np.log(
            g_sample.shape[0]
        )
    else:
        return logsumexp(err(g_sample)) - np.log(g_sample.shape[0])


def monte_carlo_error(err, p_sample):
    """
    :param err: log-error function
    :param p_sample:
    :return: log-MCE
    """
    return logsumexp(err(p_sample)) - np.log(p_sample.shape[0])


def clip(a, b_min, b_max):
    if a < b_min:
        return [np.log(b_min), 1]
    elif a > b_max:
        return [np.log(b_max), 1]
    else:
        return [np.log(a), 0]


def smooth_clip(x, eps):
    return (1 + eps) / (1 + (2 * eps / (1 - eps)) * np.exp(-x))


def ISE_clip(err, p, g, g_sample, eps, smooth_flag=True, delta = 0.1, thrhold = 0.95):
    """
    :param err: log-error function
    :param p: log-probability density of target distribution
    :param g: log-probability density of sample distribution
    :param g_sample:
    :param eps: epsilon for clip
    :return: log-ISE with clip
    """
    clipped_array = []

    log = logging.getLogger("__main__")
    if(eps == "quantile"):
        num_minmax = 0
        for eps_temp in np.arange(0, 1, delta):
            clipped_array = []       
            for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
                weight = np.exp(p_elem - g_elem)
                elem = clip(weight, eps_temp)
                num_minmax = num_minmax + elem[1]
                clipped_array.append(elem[0])
           
            if (num_minmax/g_sample.shape[0] <= thrhold):
                eps = eps_temp
            else:
                log.info(f"thrhold didnt reached, used MCE_g max value eps!")
    
    if smooth_flag:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            weight = np.exp(p_elem - g_elem)
            if np.isnan(weight):
                logging.info(f"weight is: weight")
            else:
                clipped_array.append(np.log(smooth_clip(weight, eps)))

    return logsumexp(clipped_array + err(g_sample)) - np.log(g_sample.shape[0])
