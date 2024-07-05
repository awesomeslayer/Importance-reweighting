import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from mandoline_src.mandoline import estimate_performance


def mandoline_error(g_test, p_test, model, f, err, n_slices=3):
    # D_src = slice_prediction(g_test, model, n_slices=n_slices)
    # D_tgt = slice_prediction(
    #    p_test, model, n_slices=n_slices
    # )

    D_src = slice_clusterisation(g_test, n_slices=3)
    D_tgt = slice_clusterisation(p_test, n_slices=3)

    empirical_mat_list_src = [get_correct(model, g_test, f(g_test))]
    
    est = estimate_performance(
                D_src, D_tgt, None, empirical_mat_list_src)
    
    return logsumexp(
        np.log(
           est.density_ratios
        )
        + err(g_test)
    ) - np.log(
        g_test.shape[0]
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


def slice_clusterisation(X, n_slices=3):
    # visualize_GMM_config(config, alpha):
    kmeans = KMeans(n_clusters=n_slices, n_init="auto")
    kmeans.fit(X)
    D = np.zeros((len(X), n_slices))
    for i in range(n_slices):
        for j in range(len(X)):
            if kmeans.labels_[j] == i:
                D[j][i] = 1
            else:
                D[j][i] = -1

    return D
