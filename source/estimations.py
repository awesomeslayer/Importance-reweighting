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
        return b_min, 1
    elif a > b_max:
        return b_max, 1
    else:
        return a, 0


def smooth_clip(x, eps):
    return (1 + eps) / (1 + (2 * eps / (1 - eps)) * np.exp(-x))


def ISE_clip(err, p, g, g_sample, eps, smooth_flag=True, thrhold = 0.95, clip_step = 0.001):
    """
    :param err: log-error function
    :param p: log-probability density of target distribution
    :param g: log-probability density of sample distribution
    :param g_sample:
    :param eps: epsilon for clip
    :return: log-ISE with clip
    """
    log = logging.getLogger("__main__")
    if(eps == "quantile"):
        for eps_temp in np.arange(1 - clip_step, -clip_step, -clip_step):
            clipped_array = []
            num_clipped = 0       
            for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
                weight = np.exp(p_elem - g_elem)
                clipped_weight, i = clip(weight, 1-eps_temp, 1+eps_temp)
                clipped_array.append(np.log(clipped_weight))
                num_clipped = num_clipped + i

            #log.debug(f"for eps = {eps_temp} got num = {num_clipped} instead of {len(g_sample) * thrhold}")
            if num_clipped > len(g_sample) * thrhold:
                eps = eps_temp
                log.debug(f"eps_clip_quantile = {eps}")
                break
    else:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
                weight = np.exp(p_elem - g_elem)
                clipped_weight, i = clip(weight, 1-eps_temp, 1+eps_temp)
                clipped_array.append(np.log(clipped_weight))
    
    if smooth_flag:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            weight = np.exp(p_elem - g_elem)
            clipped_array.append(np.log(smooth_clip(weight, eps)))

    return logsumexp(clipped_array + err(g_sample)) - np.log(g_sample.shape[0])
