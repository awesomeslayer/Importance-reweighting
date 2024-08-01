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

    method_list_all = params["x"] + params["x_hyp"]
    method_list_estim = [
        s for s in method_list_all if "estim" in s or "reg" in s or "MCE" in s
    ]
    method_list_no_estim = [
        s for s in method_list_all if s not in method_list_estim or s == "MCE_g"
    ]

    if "KL_LSCV" in hyperparams_params["bw_list"]:
        method_list_all += [
            "ISE_g_estim_KL",
        ]
        method_list_estim += [
            "ISE_g_estim_KL",
        ]

    errors_plot = {}
    for x in method_list_all:
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

        for x_temp in method_list_all:
            errors_plot[x_temp] += [elem["mape"][x_temp]]
    log.info(f"\nfor max_cov_list = {params['max_cov_list']}:\nerrors_plot =\n {errors_plot}\n")
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in method_list_estim:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_estim.pdf"
    )
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in method_list_no_estim:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_no_estim.pdf"
    )
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in method_list_all:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov.pdf"
    )
    plt.tight_layout()


log = logging.getLogger("__main__")
log.setLevel(logging.DEBUG)
if __name__ == "__main__":
    max_cov_plot()
