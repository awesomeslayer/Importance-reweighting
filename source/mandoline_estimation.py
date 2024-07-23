import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from mandoline_src.mandoline import log_density_ratio, mandoline


def mandoline_error(gen_dict, n_slices=3, slice_method="clusters"):
    if slice_method == "pred":
        D_src = slice_prediction(
            gen_dict["g_test"], gen_dict["model"], n_slices=n_slices
        )
        D_tgt = slice_prediction(
            gen_dict["p_test"], gen_dict["model"], n_slices=n_slices
        )

    elif slice_method == "clusters":
        kmeans = KMeans(n_clusters=n_slices, n_init="auto")
        kmeans.fit(gen_dict["g_test"])

        D_src = slice_clusterisation(gen_dict["g_test"], kmeans, n_slices=n_slices)
        D_tgt = slice_clusterisation(gen_dict["p_test"], kmeans, n_slices=n_slices)
    else:
        logging.info(f"Bad slicing_method, choose correct!")
        return False
    # Run the solver
    solved = mandoline(D_src, D_tgt, edge_list=None, sigma=1)

    # Compute the weights on the source dataset
    log_density_ratios = log_density_ratio(solved.Phi_D_src, solved)

    # print smth for max_cov ?
    return logsumexp(log_density_ratios + gen_dict["err"](gen_dict["g_test"])) - np.log(
        gen_dict["g_test"].shape[0]
    )


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


def slice_clusterisation(X, kmeans, n_slices=3):
    labels = kmeans.predict(X)
    D = np.zeros((len(labels), n_slices))
    for i in range(n_slices):
        for j in range(len(labels)):
            if labels[j] == i:
                D[j][i] = 1
            else:
                D[j][i] = -1

    return D
