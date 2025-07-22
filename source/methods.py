import logging

import numpy as np

from source.estimations import (ISE, MCE, Classifier_error, ISE_clip, ISE_deg,
                                KMM_error)

log = logging.getLogger("__main__")


class IS_method:
    # foo for at least one list in error, bw = g as default for no_estim methods (see in run)
    def __init__(self, name, n_tests, hyperparams_list=["foo"], bw_list=["g"]):
        self.name = name
        self.hyperparams_list = hyperparams_list
        self.bw_list = bw_list

        self.best_metrics_dict = {
            "mape": [],
            "rmse": [],
            "rmspe": [],
            "corr": [],
            "mape_interval": [],
            "rmse_interval": [],
            "rmspe_interval": [],
            "corr_interval": [],
        }

        self.best_hyperparams_list = []
        self.best_bw_list = []

        # for each bw, for each hyperparam init err_list with n_tests size
        # and metrics_dict with same sizes (n_tests -> one metrics (mape/rmse))
        self.test_errors_list = [
            [np.zeros(n_tests) for _ in range(len(hyperparams_list))]
            for _ in range(len(bw_list))
        ]

        self.test_metrics_dict = {
            "mape": [np.zeros(len(hyperparams_list)) for _ in range(len(bw_list))],
            "mape_interval": [
                np.empty(len(hyperparams_list), dtype=object)
                for _ in range(len(bw_list))
            ],
            "rmse": [np.zeros(len(hyperparams_list)) for _ in range(len(bw_list))],
            "rmse_interval": [
                np.empty(len(hyperparams_list), dtype=object)
                for _ in range(len(bw_list))
            ],
            "rmspe": [np.zeros(len(hyperparams_list)) for _ in range(len(bw_list))],
            "rmspe_interval": [
                np.empty(len(hyperparams_list), dtype=object)
                for _ in range(len(bw_list))
            ],
            "corr": [np.zeros(len(hyperparams_list)) for _ in range(len(bw_list))],
            "corr_interval": [
                np.empty(len(hyperparams_list), dtype=object)
                for _ in range(len(bw_list))
            ],
        }
        log.debug(f"inited with name = {name}")

    # default names: "ISE_deg", "ISE_uni", "ISE_estim", "ISE_estim_clip", "MCE_g", "ISE", "ISE_clip", "MCE_p"
    def single_test(self, conf, test_gen_dict, hyperparam, hyp_params_dict):
        if self.name == "MCE_p":
            error = MCE(test_gen_dict["err"], test_gen_dict["p_test"])
            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if self.name == "MCE_g":
            error = MCE(test_gen_dict["err"], test_gen_dict["g_test"])
            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if self.name == "Classifier":
            error = Classifier_error(
                test_gen_dict["g_train"],
                test_gen_dict["p_train"],
                test_gen_dict["g_test"],
                test_gen_dict["p_test"],
                test_gen_dict["err"],
            )
            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if "deg" in self.name:
            error = ISE_deg(
                test_gen_dict["err"],
                test_gen_dict["p"],
                test_gen_dict["g"],
                test_gen_dict["g_test"],
                hyperparam,
            )
            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if "uni" in self.name:
            g_estim_uni = lambda X: (1 - hyperparam) * test_gen_dict["g"](
                X
            ) + hyperparam / (conf["max_mu"] ** 2)
            error = ISE(
                test_gen_dict["err"],
                test_gen_dict["p"],
                g_estim_uni,
                test_gen_dict["g_test"],
            )
            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if "clip" in self.name:
            error = ISE_clip(
                test_gen_dict["err"],
                test_gen_dict["p"],
                test_gen_dict["g"],
                test_gen_dict["g_test"],
                hyperparam,
                smooth_flag=hyp_params_dict["smooth_flag"],
                thrhold=hyp_params_dict["clip_thrhold"],
                clip_step=hyp_params_dict["clip_step"],
            )

            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if (
            self.name == "ISE"
            or self.name == "ISE_estim"
            or self.name == "ISE_estim_KL"
        ):
            error = ISE(
                test_gen_dict["err"],
                test_gen_dict["p"],
                test_gen_dict["g"],
                test_gen_dict["g_test"],
            )
            log.debug(f"name = {self.name}, error = {error} ]")
            return error

        if self.name == "KMM":
            error = KMM_error(
                test_gen_dict["err"],
                test_gen_dict["p_test"],
                test_gen_dict["g_test"],
                hyperparam,
            )
            log.debug(f"name = {self.name}, error = {error} ]")
            return error
        return False
