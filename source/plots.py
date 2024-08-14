import matplotlib.pyplot as plt


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
            f"./plots/results/extr_plots/{params['model']}_{params['f']}/{x_method.name}/{conf['max_cov']}_bw_{x_method.bw_list[n_bw]}.pdf"
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
            f"./plots/results/bw_plots/{params['model']}_{params['f']}/{x_method.name}/{conf['max_cov']}.pdf"
        )
        plt.tight_layout()

    return True
