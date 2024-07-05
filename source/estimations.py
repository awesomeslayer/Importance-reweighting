import numpy as np
from scipy.special import logsumexp


def importance_sampling_error(err, p, g, g_sample):
    """
    :param err: log-error function
    :param p: log-probability density of target distribution
    :param g: log-probability density of sample distribution
    :param g_sample:
    :return: log-ISE
    """
    #print(f"error IS:{logsumexp(p(g_sample) - g(g_sample) + err(g_sample))}")
    return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )


def monte_carlo_error(err, p_sample):
    """
    :param err: log-error function
    :param p_sample:
    :return: log-MCE
    """
    #print(f"True error Mce_p/Mce_g:{logsumexp(err(p_sample))}")
    return logsumexp(err(p_sample)) - np.log(p_sample.shape[0])


def clip(a, b_min, b_max):
    if a < b_min:
        return b_min
    elif a > b_max:
        return b_max
    else:
        return a


def ISE_clip(err, p, g, g_sample, eps):
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
        clipped_array.append(np.log(clip(np.exp(p_elem - g_elem), 1 - eps, 1 + eps)))

    return logsumexp(clipped_array + err(g_sample)) - np.log(g_sample.shape[0])


def monte_carlo_error_variance(err, p_sample):
    """
    :param err: log-error function
    :param p_sample
    :return: log-expected part of variance
    """
    return logsumexp(2 * err(p_sample)) - np.log(p_sample.shape[0])


def importance_sampling_error_variance(err, p, g, g_sample):
    """
    :param err: log-error function
    :param p: log-probability density of target distribution
    :param g: log-probability density of sample distribution
    :param g_sample:
    :return: log-expected part of variance
    """
    return logsumexp(2 * err(g_sample) + 2 * p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )
