import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from mandoline_src.mandoline import estimate_performance


def mandoline_error(g_test, p_test, model, f, err, n_slices=3):
    D_src = slice_prediction(p_test, model, n_slices=n_slices)
    D_tgt = slice_prediction(
        g_test, model, n_slices=n_slices
    )  # D_tgt has shape (n_tgt x n_slices)

    empirical_mat_list_src = [get_correct(model, g_test, f(g_test))]

    return logsumexp(
        np.log(
            estimate_performance(
                D_src, D_tgt, np.array([(0, 1), (1, 2), (0, 2)]), empirical_mat_list_src
            ).density_ratios
        )
        + err(g_test)
    ) - np.log(
        g_test.shape[0]
    )  # logsumexp(err(g_test) + log_weights) - np.log(g_test.shape[0])


def get_correct(model, X, labels, tolerance=1e-5):
    """
    Returns whether the model makes a correct prediction for each example in X.
    """
    correct = []
    for X_batch, labels_batch in zip(X, labels):
        preds = model.predict([X_batch])
        correct.extend(np.abs(preds - labels_batch) < tolerance)

    return np.array(correct)[:, np.newaxis]  # (n, 1) binary np.ndarray


def slice_prediction(X, model, n_slices=2):
    left = min(model.predict(X))
    right = max(model.predict(X))
    delta = np.abs(right - left) / n_slices
    D = np.zeros((len(X), n_slices))

    for i in range(n_slices):
        for j, x in enumerate(X):
            if (
                model.predict([x]) > left + i * delta
                and model.predict([x]) < left + (i + 1) * delta
            ):  # prediction is in k-bound
                D[j][i] = 1
            else:
                D[j][i] = -1
    return D


def slice_clastorisation():
    # Use K-means?
    return 0
