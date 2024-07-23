import matplotlib.pyplot as plt
import numpy as np


def extr_plots(conf, params, metrics_hyp_dict, hyperparams_dict, bw):
    for x_temp in params["x_hyp"]:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(
            hyperparams_dict[x_temp],
            metrics_hyp_dict["mape"][x_temp],
            label=f"{x_temp}",
        )
        plt.legend(fontsize=26)
        plt.title(
            f"max_cov{conf['max_cov']}_gens_{params['model']}_{params['f']}_bw{bw}"
        )
        ax.set_xlabel(f"param for {x_temp}", fontsize=26)
        ax.set_ylabel("mape", fontsize=26)
        if params["log_flag"]:
            ax.set_xscale("log")
        plt.savefig(
            f"./plots/results/extr_plots/{params['model']}_{params['f']}_{x_temp}_{conf['max_cov']}_bw{bw}.pdf"
        )
        plt.tight_layout()
        # plt.show()


def bw_plot(conf, params, bw_list, best_metrics_dict):
    if len(bw_list) > 1:
        fig, ax = plt.subplots(figsize=(12, 12))
        x_list = list(set(params["x"]) & set(["ISE_g_estim"])) + list(
            set(params["x_hyp"])
            & set(["ISE_g_estim_clip", "ISE_g_reg_uniform", "ISE_g_reg_degree"])
        )
        for x_temp in x_list:
            ax.plot(
                bw_list,
                best_metrics_dict["mape"][x_temp],
                label=f"{x_temp}",
            )

        plt.legend(fontsize=26)
        plt.title(
            f"max_cov{conf['max_cov']}_gens_{params['model']}_max_cov{conf['max_cov']}."
        )
        ax.set_xlabel(f"bandwidth", fontsize=26)
        ax.set_ylabel("mape", fontsize=26)
        plt.savefig(
            f"./plots/results/bw_plots/{params['model']}_{params['f']}_max_cov{conf['max_cov']}.pdf"
        )
        plt.tight_layout()
        # plt.show()

    for x_temp, er_list in best_metrics_dict["mape"].items():
        print(x_temp, er_list)
        best_metrics_dict["mape"][x_temp] = er_list[er_list.index(min(er_list))]
        best_metrics_dict["rmse"][x_temp] = best_metrics_dict["rmse"][x_temp][
            er_list.index(min(er_list))
        ]

    return best_metrics_dict
