import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange
from mandoline_src.mandoline import mandoline, log_density_ratio


def mandoline_error(g_test, p_test, model, f, err, n_slices=3):
    # for prediction-based slices:
    # D_src = slice_prediction(g_test, model, n_slices=n_slices)
    # D_tgt = slice_prediction(
    #    p_test, model, n_slices=n_slices
    # )
    kmeans = KMeans(n_clusters=n_slices, n_init="auto")
    kmeans.fit(g_test)

    D_src = slice_clusterisation(g_test, kmeans, n_slices=n_slices)
    D_tgt = slice_clusterisation(p_test, kmeans, n_slices=n_slices)
    
     # Run the solver
    solved = mandoline(D_src, D_tgt, edge_list = None, sigma = 1)

    # Compute the weights on the source dataset
    density_ratios = np.e ** log_density_ratio(solved.Phi_D_src, solved)

    # print smth for max_cov ?
    return logsumexp(np.log(density_ratios) + err(g_test)) - np.log(g_test.shape[0])


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
