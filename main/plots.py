import matplotlib.pyplot as plt
import logging
import numpy as np
from functools import partial
from tqdm import trange, tqdm
from KL_divergence_estimators.knn_divergence import (
    naive_estimator,
    scipy_estimator,
    skl_efficient,
    skl_estimator,
)
from source.KL_LSCV import squared_error
from source.LCF import (
    test_LCF,
    estimate_lcf,
    estimate_point_counts,
    compute_normalized_auc,
)
from source.simulation import (
    random_GMM_samples,
    random_uniform_samples,
)

log = logging.getLogger("__main__")
log.setLevel(logging.DEBUG)


def plot_cov_KL_estim(conf, params, KL_estim_list=["naive", "scipy", "skl", "skl_ef"]):
    log.debug(f"KL_estim_list = {KL_estim_list}")

    KL_estim_func_dict = {
        "naive": lambda g, p: naive_estimator(g, p, k=5),
        "scipy": lambda g, p: scipy_estimator(g, p, k=5),
        "skl": lambda g, p: skl_estimator(g, p, k=5),
        "skl_ef": lambda g, p: skl_efficient(g, p, k=5),
    }

    KL_estim_dict = {}
    for key in KL_estim_list:
        KL_estim_dict[key] = np.zeros(len(params["max_cov_list"]))

    log.debug(f"KL_estim(max_cov) plot:")
    for j, cov in tqdm(enumerate(params["max_cov_list"])):
        conf["max_cov"] = cov
        g_gen = partial(random_GMM_samples, conf)
        p_gen = partial(random_uniform_samples, conf, True)
        log.debug(f"cov = {cov}:")
        for _ in trange(params["n_tests"]):
            g_sample, _ = g_gen()
            p_sample, _ = p_gen()

            for key in KL_estim_list:
                KL_estim_dict[key][j] = (
                    KL_estim_dict[key][j]
                    + KL_estim_func_dict[key](g_sample, p_sample) / params["n_tests"]
                )

    fig, ax = plt.subplots(figsize=(12, 12))
    for key in KL_estim_dict:
        ax.plot(
            params["max_cov_list"],
            KL_estim_dict[key],
            label=f"{key}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("KL_estimation between u and g", fontsize=26)
    plt.tight_layout()
    plt.savefig(
        f"./main/results/KL_plots/{params['model']}_{params['f']}/KL_estim(max_cov).pdf"
    )
    return True


def plot_cov_LCF(conf, params, r_values=np.arange(0, 50, 1)):
    test_LCF()

    LCF_dict = {"p": [], "g": []}
    for key in LCF_dict:
        LCF_dict[key] = np.zeros(len(params["max_cov_list"]))

    log.debug(f"LCF(max_cov) plot:")
    for j, cov in tqdm(enumerate(params["max_cov_list"])):
        conf["max_cov"] = cov
        g_gen = partial(random_GMM_samples, conf)
        p_gen = partial(random_uniform_samples, conf, True)
        log.debug(f"cov = {cov}:")
        for _ in trange(params["n_tests"]):
            g_sample, _ = g_gen()
            p_sample, _ = p_gen()

            p_est_df = estimate_point_counts(p_sample, r_values)
            g_est_df = estimate_point_counts(g_sample, r_values)

            p_lcf_df = estimate_lcf(p_est_df, r=r_values)
            g_lcf_df = estimate_lcf(g_est_df, r=r_values)

            AUC_p = compute_normalized_auc(p_lcf_df, r_values)
            AUC_g = compute_normalized_auc(g_lcf_df, r_values)

            for key, LCF_AUC in zip(LCF_dict, (AUC_p, AUC_g)):
                LCF_dict[key][j] = LCF_dict[key][j] + LCF_AUC / params["n_tests"]

    fig, ax = plt.subplots(figsize=(12, 12))
    for key in LCF_dict:
        ax.plot(
            params["max_cov_list"],
            LCF_dict[key],
            label=f"{key}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("LCF_AUC_average", fontsize=26)
    plt.title(f"n_test = {params['n_tests']}")
    plt.tight_layout()
    plt.savefig(f"./main/results/LCF_plots/LCF(max_cov).pdf")
    return True


def plot_KL_bw(
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
