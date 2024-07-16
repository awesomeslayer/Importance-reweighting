import numpy as np
from scipy.special import logsumexp


# def importance_sampling_error(err, p, g, g_sample):
#     """
#     :param err: log-error function
#     :param p: log-probability density of target distribution
#     :param g: log-probability density of sample distribution
#     :param g_sample:
#     :return: log-ISE
#     """
#     # print(f"error IS:{logsumexp(p(g_sample) - g(g_sample) + err(g_sample))}")
#     return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
#         g_sample.shape[0]
#     )


def importance_sampling_error(err, p, g, g_sample):
    sum = 0
    for i in range(g_sample.shape[0]):
        sum = sum + err([g_sample[i]]) * p([g_sample[i]]) / g([g_sample[i]])
    return np.squeeze(sum) / g_sample.shape[0]


def monte_carlo_error(err, p_sample):
    """
    :param err: log-error function
    :param p_sample:
    :return: log-MCE
    """
    sum = 0
    for i in range(p_sample.shape[0]):
        sum = sum + err([p_sample[i]])
    return np.squeeze(sum)/p_sample.shape[0]


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
    if smooth_flag:
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            clipped_array.append(np.log(smooth_clip(p_elem/g_elem, eps)))
    else:
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            clipped_array.append(
                np.log(clip(p_elem/g_elem, 1 - eps, 1 + eps))
            )
    sum = 0
    
    for i in range(g_sample.shape[0]):
        sum = sum + clipped_array[i]*err([g_sample[i]])
    return np.squeeze(sum)/g_sample.shape[0]
