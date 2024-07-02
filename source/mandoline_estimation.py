import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from mandoline.mandoline import estimate_performance


def mandoline_error(g_train, g_test, model, f):

    D_src = slice_prediction(g_train)
    D_tgt = slice_prediction(g_test)  # D_tgt has shape (n_tgt x n_slices)

    empirical_mat_list_src = [get_correct(model, g_train, f(g_train))]
    print("emperical mat list src")
    print(empirical_mat_list_src)

    estimate_performance(D_src, D_tgt, None, empirical_mat_list_src)

    # D_src = slice_clastorisation(g_test)
    # D_tgt = slice_clastorisation(g_train)

    # estimate_performance(D_src, D_tgt, None, emerical_mat_list_src)

    return 0


def get_correct(model, X, labels):
    """
    Returns whether the model makes a correct prediction for each example in X.
    """
    correct = []
    for X_batch, labels_batch in zip(X, labels):
        preds = model.predict(X_batch)
        correct.extend(preds == labels_batch)

    return np.array(correct)[:, np.newaxis]  # (n, 1) binary np.ndarray


def slice_prediction(X, model, n_slices):
    left = min(model.predict(X))
    right = max(model.predict(X))
    print(f"left = {left}")
    delta = np.abs(right - left)
    print(f"delta = {delta}")
    D = np.zeros(len(X), n_slices)
    for i in range(n_slices):
        for j, x in enumerate(X):
            if (
                model.predict(x) > left + i * delta
                and model.predict(x) < left + (i + 1) * delta
            ):  # prediction is in k-bound
                D[j][i] = 1
            else:
                D[j][i] = -1
    print(f"D:\n{D}")
    return D


def slice_clastorisation():
    return 0
