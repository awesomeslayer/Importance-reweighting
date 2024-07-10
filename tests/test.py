import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.stats import gaussian_kde
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from source.estimations import (
    importance_sampling_error,
    importance_sampling_error_default,
    monte_carlo_error,
    ISE_clip,
)

from source.mandoline_estimation import mandoline_error


def test(
    conf,
    f_gen,
    model,
    g_gen,
    p_gen,
    n_tests,
    n_splits=1,
    target_error=None,
    hyperparams={
        "kde_size": 5,
        "n_slices": 3,
        "ISE_g_regular": 0,
        "ISE_g_clip": 0,
        "ISE_g_estim_clip": 0,
    },
):

    estimation_list = [
        "MCE_p",
        "MCE_g",
        "ISE_g",
        "ISE_g_estim",
        "ISE_g_regular",
        "ISE_g_clip",
        "ISE_g_estim_clip",
        "Mandoline",
    ]

    if target_error is None:
        target_error = estimation_list

    if isinstance(target_error, str):
        target_error = [target_error]

    for err in target_error:
        if err not in estimation_list:
            print(err)
            raise KeyError

    CV_err = dict()
    for err in target_error:
        CV_err[err] = []

    kf = (
        KFold(n_splits=n_splits)
        if n_splits > 1
        else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    )

    for _ in range(n_tests):
        iter_err = dict()
        for err in target_error:
            iter_err[err] = []

        f = f_gen()
        g_sample, g = g_gen()
        p_sample, p = p_gen()

        for i, (train_idx, test_idx) in enumerate(kf.split(g_sample)):

            g_train = g_sample[train_idx]
            g_test = g_sample[test_idx]
            p_test = p_sample[test_idx]

            model.fit(g_train, f(g_train))

            err = lambda X: np.log(np.abs(f(X) - model.predict(X)))

            kde_list = [
                "ISE_g_estim",
                "ISE_g_regular",
                "ISE_g_estim_clip",
                "ISE_g_estim_variance",
                "ISE_g_regular_variance",
            ]
            if [i for i in target_error if i in kde_list]:
                kde_sk = KernelDensity(
                    kernel="gaussian", bandwidth=hyperparams["kde_size"]
                ).fit(g_train)
                g_estim = lambda X: kde_sk.score_samples(X)

            if "MCE_p" in target_error:

                iter_err["MCE_p"] += [monte_carlo_error(err, p_test)]

            if "MCE_g" in target_error:

                iter_err["MCE_g"] += [monte_carlo_error(err, g_test)]

            if "ISE_g" in target_error:

                iter_err["ISE_g"] += [importance_sampling_error(err, p, g, g_test)]

            if "ISE_g_estim" in target_error:
                iter_err["ISE_g_estim"] += [
                    importance_sampling_error_default(
                        lambda X: np.exp(err(X)),
                        lambda X: np.exp(p(X)),
                        lambda X: np.exp(g_estim(X)),
                        g_test,
                    )
                ]

            if "ISE_g_regular" in target_error:
                epsilon = hyperparams["ISE_g_regular"]
                g_estim_new = lambda X: (1 - epsilon) * np.exp(g_estim(X)) + epsilon / (
                    conf["max_mu"] ** 2
                )
                if epsilon == 0:
                    logging.debug("Comparing to the truth for eps = 0:")
                    delta = np.abs(
                        (
                            importance_sampling_error_default(
                                lambda X: np.exp(err(X)),
                                lambda X: np.exp(p(X)),
                                g_estim_new,
                                g_test,
                            )
                            - importance_sampling_error_default(
                                lambda X: np.exp(err(X)),
                                lambda X: np.exp(p(X)),
                                lambda X: np.exp(g_estim(X)),
                                g_test,
                            )
                        )[0]
                    )
                    logging.debug(f"delta_errors:{delta}")
                iter_err["ISE_g_regular"] += [
                    importance_sampling_error_default(
                        lambda X: np.exp(err(X)),
                        lambda X: np.exp(p(X)),
                        g_estim_new,
                        g_test,
                    )
                ]

            if "ISE_g_clip" in target_error:
                iter_err["ISE_g_clip"] += [
                    ISE_clip(err, p, g, g_test, hyperparams["ISE_g_clip"])
                ]

            if "ISE_g_estim_clip" in target_error:
                iter_err["ISE_g_estim_clip"] += [
                    ISE_clip(err, p, g_estim, g_test, hyperparams["ISE_g_estim_clip"])
                ]

            if "Mandoline" in target_error:
                iter_err["Mandoline"] += [
                    mandoline_error(
                        g_test, p_test, model, f, err, n_slices=hyperparams["Mandoline"]
                    )
                ]

        for err in target_error:
            CV_err[err] += [logsumexp(iter_err[err]) - np.log(n_splits)]

    return CV_err
