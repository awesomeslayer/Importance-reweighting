import logging

import hydra
import matplotlib.pyplot as plt
import tqdm
from omegaconf import DictConfig, OmegaConf

from source.run import run


@hydra.main(version_base=None, config_path="../config", config_name="config")
def max_cov_plot(cfg: DictConfig):
    hyperparams_dict = OmegaConf.to_container(cfg["hyperparams_dict"])
    params = OmegaConf.to_container(cfg["params"])
    hyperparams_params = OmegaConf.to_container(cfg["hyperparams_params"])
    conf = OmegaConf.to_container(cfg["conf"])

    methods_list = params["x"] + params["x_hyp"]
    if "KL_LSCV" in hyperparams_params["bw_list"]:
        methods_list += [
            "ISE_g_estim_KL",
            "ISE_g_reg_uniform_KL",
            "ISE_g_estim_clip_KL",
            "ISE_g_reg_degree_KL",
        ]

    errors_plot = {}
    for x in methods_list:
        errors_plot[x] = []

    log.info(f"\nparams: {params}\n")
    log.info(f"\nconfig: {conf}\n")
    log.info(f"\nhyperparams_dict; {hyperparams_dict}\n")
    log.info(f"\nhyperparams_params: {hyperparams_params}\n")

    for max_cov in tqdm.tqdm(params["max_cov_list"]):
        log.info(f"max_cov:{max_cov}")
        conf["max_cov"] = max_cov

        elem = run(conf, params, hyperparams_params, hyperparams_dict)

        log.info(f"\nmape's and rmse's for max_cov={max_cov}:\n {elem}\n")

        for x_temp in methods_list:
            errors_plot[x_temp] += [elem["mape"][x_temp]]

    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in ["MCE_g", "ISE_g", "ISE_g_clip", "Mandoline"]:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_emp.pdf"
    )
    plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in [
        "MCE_g",
        "ISE_g_estim",
        "ISE_g_reg_uniform",
        "ISE_g_reg_degree",
        "ISE_g_estim_clip",
        "Mandoline",
    ]:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_estim.pdf"
    )
    plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in methods_list:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov.pdf"
    )
    plt.tight_layout()
    # plt.show()


log = logging.getLogger("__main__")
log.setLevel(logging.DEBUG)
if __name__ == "__main__":
    max_cov_plot()
