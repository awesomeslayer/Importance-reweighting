import logging

import numpy as np
from scipy.special import logsumexp


def importance_sampling_error(err, p, g, g_sample):
    return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )


def monte_carlo_error(err, p_sample):
    """
    :param err: log-error function
    :param p_sample:
    :return: log-MCE
    """
    return logsumexp(err(p_sample)) - np.log(p_sample.shape[0])


def clip(a, b_min, b_max):
    if a < b_min:
        return b_min
    elif a > b_max:
        return b_max
    else:
        return a


def smooth_clip(x, eps):
    return (1 + eps) / (1 + (2 * eps / (1 - eps)) * np.exp(-x))


def ISE_clip(err, p, g, g_sample, eps, smooth_flag=True):
    """
    :param err: log-error function
    :param p: log-probability density of target distribution
    :param g: log-probability density of sample distribution
    :param g_sample:
    :param eps: epsilon for clip
    :return: log-ISE with clip
    """
    clipped_array = []
    for p_elem, g_elem in zip(p(g_sample), g(g_sample)):

        weight = np.exp(p_elem - g_elem)
        if np.isnan(weight):
            logging.info(weight)
        if smooth_flag:
            clipped_array.append(np.log(smooth_clip(weight, eps)))
        else:
            clipped_array.append(np.log(clip(weight, 1 - eps, 1 + eps)))

    return logsumexp(clipped_array + err(g_sample)) - np.log(g_sample.shape[0])
