import matplotlib.pyplot as plt
from .plots import plot_cov_KL_estim
from .read_configs import read_configs
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from source.run import run


@hydra.main(version_base=None, config_path="../config", config_name="config")
def plot_max_cov(cfg: DictConfig):

    conf, params, methods_list, hyp_params_dict = read_configs(cfg)
    plot_cov_KL_estim(conf, params)

    for max_cov in tqdm(params["max_cov_list"]):
        conf["max_cov"] = max_cov
        run(conf, params, methods_list, hyp_params_dict)

    methods_list_all = [method for method in methods_list if method.name != "MCE_p"]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_all:
        ax.plot(
            params["max_cov_list"],
            x_method.best_metrics_dict["mape"],
            label=f"{x_method.name}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./main/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_all.pdf"
    )
    plt.tight_layout()

    methods_list_no_estim = [
        method for method in methods_list if method.name in ["MCE_g", "ISE", "ISE_clip"]
    ]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_no_estim:
        ax.plot(
            params["max_cov_list"],
            x_method.best_metrics_dict["mape"],
            label=f"{x_method.name}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./main/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_no_estim.pdf"
    )
    plt.tight_layout()

    methods_list_estim = [
        method
        for method in methods_list
        if method.name in ["MCE_g", "ISE_deg", "ISE_uni", "ISE_estim", "ISE_estim_clip"]
    ]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_estim:
        ax.plot(
            params["max_cov_list"],
            x_method.best_metrics_dict["mape"],
            label=f"{x_method.name}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./main/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_estim.pdf"
    )
    plt.tight_layout()

    methods_list_KL = [
        method
        for method in methods_list
        if method.name
        in ["MCE_g", "ISE_deg_KL", "ISE_uni_KL", "ISE_estim_KL", "ISE_estim_clip_KL"]
    ]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_KL:
        ax.plot(
            params["max_cov_list"],
            x_method.best_metrics_dict["mape"],
            label=f"{x_method.name}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./main/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_KL.pdf"
    )
    plt.tight_layout()

    return True


if __name__ == "__main__":
    plot_max_cov()
