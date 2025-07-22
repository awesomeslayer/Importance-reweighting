<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
from .plots import plot_cov_KL_estim, plot_cov_LCF, plot_cov_KS, plot_cov_bw
from .read_configs import read_configs
>>>>>>> c25b52be1e00f0f64060edceaf4404d54a55df45
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from source.run import run

<<<<<<< HEAD
from .plots import plot_cov_bw, plot_cov_KL_estim, plot_cov_KS, plot_cov_LCF
from .read_configs import read_configs
=======
>>>>>>> c25b52be1e00f0f64060edceaf4404d54a55df45


@hydra.main(version_base=None, config_path="../config", config_name="config")
def plot_max_cov(cfg: DictConfig):

    conf, params, methods_list, hyp_params_dict = read_configs(cfg)
<<<<<<< HEAD

    if hyp_params_dict["metrics_plots"]:
        plot_cov_LCF(conf, params)
        plot_cov_KS(conf, params, n_tests=30)

=======
        
    if hyp_params_dict["metrics_plots"]:
        plot_cov_LCF(conf, params)
        plot_cov_KS(conf, params, n_tests=30)
    
>>>>>>> c25b52be1e00f0f64060edceaf4404d54a55df45
        plot_cov_bw(conf, params, hyp_params_dict)
        plot_cov_KL_estim(conf, params, ["skl"])

    for max_cov in tqdm(params["max_cov_list"]):
        conf["max_cov"] = max_cov
        run(conf, params, methods_list, hyp_params_dict)

    methods_list_all = []
    methods_list_all += [[method for method in methods_list if method.name != "MCE_p"]]
    methods_list_all += [
        [
            method
            for method in methods_list
            if method.name in ["MCE_g", "ISE", "ISE_clip"]
        ]
    ]
    methods_list_all += [
        [
            method
            for method in methods_list
            if method.name
            in ["MCE_g", "ISE_deg", "ISE_uni", "ISE_estim", "ISE_estim_clip"]
        ]
    ]
    methods_list_all += [
        [
            method
            for method in methods_list
            if method.name
            in [
                "MCE_g",
                "ISE_deg_KL",
                "ISE_uni_KL",
                "ISE_estim_KL",
                "ISE_estim_clip_KL",
            ]
        ]
    ]

    for methods_list, name in zip(methods_list_all, ["all", "no_estim", "estim", "KL"]):
        fig, ax = plt.subplots(figsize=(12, 12))
        for x_method in methods_list:
            mape = x_method.best_metrics_dict["mape"]
            mape_interval = [
                bound[1] - bound[0]
                for bound in x_method.best_metrics_dict["mape_interval"]
            ]
<<<<<<< HEAD
            if hyp_params_dict["errorbar_flag"]:
=======
            if(hyp_params_dict["errorbar_flag"]):
>>>>>>> c25b52be1e00f0f64060edceaf4404d54a55df45
                ax.errorbar(
                    params["max_cov_list"],
                    mape,
                    yerr=mape_interval,
                    label=f"{x_method.name}",
                    fmt="-o",
                    capsize=5,
                    elinewidth=2,
                )
            else:
<<<<<<< HEAD
                ax.plot(params["max_cov_list"], mape, label=f"{x_method.name}")
=======
                ax.plot(params["max_cov_list"], mape, label = f"{x_method.name}")
>>>>>>> c25b52be1e00f0f64060edceaf4404d54a55df45
        plt.legend(fontsize=26)
        ax.set_xlabel("clusters size", fontsize=26)
        ax.set_ylabel("mape(true_risk, estimated_risk)", fontsize=26)
        plt.savefig(
            f"./main/results/max_cov_plots/{params['samples']}/{params['model']}_{params['f']}_max_cov_{name}.pdf"
        )

        plt.tight_layout()

    return True


if __name__ == "__main__":
    plot_max_cov()
