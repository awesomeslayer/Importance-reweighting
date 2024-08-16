import matplotlib.pyplot as plt
import logging
import numpy as np
from functools import partial
from KL_divergence_estimators.knn_divergence import (
    naive_estimator,
    scipy_estimator,
    skl_efficient,
    skl_estimator,
)
from source.KL_LSCV import squared_error
from source.simulation import (
    random_GMM_samples,
    random_uniform_samples,
)

log = logging.getLogger("__main__")
log.setLevel(logging.DEBUG)


def plot_cov_KL_estim(conf, cov_list, n_tests):
    KL_estim_dict = {
        "naive": np.zeros(len(cov_list)),
        "scipy": np.zeros(len(cov_list)),
        "skl": np.zeros(len(cov_list)),
        "skl_ef": np.zeros(len(cov_list)),
    }

    for j, cov in enumerate(cov_list):
        conf["max_cov"] = cov
        g_gen = partial(random_GMM_samples, conf)
        p_gen = partial(random_uniform_samples, conf, True)
        for i in range(n_tests):
            g_sample, g = g_gen()
            p_sample, p = p_gen()
            print(f"test = {i}")
            KL_estim_dict["naive"][j] = (
                KL_estim_dict["naive"][j]
                + naive_estimator(g_sample, p_sample, k=5) / n_tests
            )
            KL_estim_dict["scipy"][j] = (
                KL_estim_dict["scipy"][j]
                + scipy_estimator(g_sample, p_sample, k=5) / n_tests
            )
            KL_estim_dict["skl"][j] = (
                KL_estim_dict["skl"][j]
                + skl_estimator(g_sample, p_sample, k=5) / n_tests
            )
            KL_estim_dict["skl_ef"][j] = (
                KL_estim_dict["skl_ef"][j]
                + skl_efficient(g_sample, p_sample, k=5) / n_tests
            )

    fig, ax = plt.subplots(figsize=(12, 12))
    for key in KL_estim_dict:
        ax.plot(
            cov_list,
            KL_estim_dict[key],
            label=f"{key}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("KL_estimation between u and g", fontsize=26)
    plt.tight_layout()
    plt.show()
    return True


def plot_KL(
    conf,
    g_sample,
    p_sample,
    beta,
    flag,
    estim_type,
    params,
    h_list=np.arange(1, 30, 0.33),
):
    kl_list = []
    for h in h_list:
        kl_list += [
            squared_error([h], conf, g_sample, p_sample, beta, flag, estim_type)
        ]

    log.debug(f"h_list:{h_list}")
    log.debug(f"kl_list:{kl_list}")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(
        h_list,
        kl_list,
        label=f"KL",
    )
    plt.legend(fontsize=26)
    plt.title(f"KL(h) for {conf['max_cov']}")
    ax.set_xlabel("h", fontsize=26)
    ax.set_ylabel("KL", fontsize=26)
    plt.tight_layout()
    plt.savefig(
        f"./main/results/KL_plots/{params['model']}_{params['f']}/cov_{conf['max_cov']}.pdf"
    )
    return True


def plot_extr_hyp(conf, params, x_method, n_bw):

    if len(x_method.hyperparams_list) > 1:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(
            x_method.hyperparams_list,
            x_method.test_metrics_dict["mape"][n_bw],
            label=f"{x_method.name}",
        )
        plt.legend(fontsize=26)
        plt.title(f"max_cov{conf['max_cov']}_bw{x_method.bw_list[n_bw]}")
        ax.set_xlabel(f"param for {x_method.name}", fontsize=26)
        ax.set_ylabel("mape", fontsize=26)

        plt.savefig(
            f"./main/results/extr_plots/{params['model']}_{params['f']}/{x_method.name}/{conf['max_cov']}_bw_{x_method.bw_list[n_bw]}.pdf"
        )
        plt.tight_layout()

    return True


def plot_extr_bw(conf, params, best_metrics_hyp, x_method):
    if len(x_method.bw_list) > 1:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(
            x_method.bw_list,
            best_metrics_hyp,
            label=f"{x_method.name}",
        )
        plt.legend(fontsize=26)
        plt.title(f"max_cov{conf['max_cov']}")
        ax.set_xlabel(f"bw", fontsize=26)
        ax.set_ylabel("best_mape for bw", fontsize=26)
        plt.savefig(
            f"./main/results/bw_plots/{params['model']}_{params['f']}/{x_method.name}/{conf['max_cov']}.pdf"
        )
        plt.tight_layout()

    return True
