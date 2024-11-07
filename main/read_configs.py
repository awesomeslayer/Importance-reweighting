import logging
from copy import copy
from omegaconf import OmegaConf
from source.methods import IS_method
import numpy as np

log = logging.getLogger("__main__")


def read_configs(cfg):
    conf = OmegaConf.to_container(cfg["conf"])
    params = OmegaConf.to_container(cfg["params"])
    methods_names = OmegaConf.to_container(cfg["methods"])
    hyp_dict = OmegaConf.to_container(cfg["hyp_dict"])
    hyp_params_dict = OmegaConf.to_container(cfg["hyp_params_dict"])
    
    hyp_dict["ISE_uni"] = np.linspace(0, 1, 100)
    hyp_dict["ISE_deg"] = np.linspace(0, 3, 100)
    hyp_dict["ISE_clip"] = np.linspace(0.0001, 0.1, 50) + np.linspace(0.1, 1, 50)

    log.info(
        f"Starting configs:\n conf:\n{conf},\n params:\n{params},\n method_names:\n{methods_names},\n hyp_dict:\n{hyp_dict},\n hyp_params_dict:\n{hyp_params_dict}"
    )

    bw_dict = {}
    if "KL" in hyp_params_dict["bw_list"]:
        hyp_params_dict["bw_list"].remove("KL")

        for s in copy(methods_names["x_estim"]):
            methods_names["x_estim"] += [s + "_KL"]
            bw_dict[s] = copy(hyp_params_dict["bw_list"])

            bw_dict[s + "_KL"] = ["KL"]

            if s in hyp_dict:
                hyp_dict[s + "_KL"] = hyp_dict[s]
            else:
                hyp_dict[s] = ["foo"]
                hyp_dict[s + "_KL"] = ["foo"]

        hyp_params_dict["bw_list"].append("KL")

    else:
        for s in methods_names["x_estim"]:
            bw_dict[s] = copy(hyp_params_dict["bw_list"])
            if s not in hyp_dict:
                hyp_dict[s] = ["foo"]

    for s in methods_names["x_no_estim"] + methods_names["y"]:
        bw_dict[s] = ["g"]
        if s not in hyp_dict:
            hyp_dict[s] = ["foo"]

    methods_names_all = methods_names["x_estim"] + methods_names["x_no_estim"]
    print(
        f"Filled configs for methods:\n method_names:\n{methods_names_all},\n hyp_dict:\n{hyp_dict},\n bw_dict:\n{bw_dict}"
    )

    methods_list = []
    for name in methods_names_all + methods_names["y"]:
        methods_list += [
            IS_method(name, params["n_tests"], hyp_dict[name], bw_dict[name])
        ]
    return conf, params, methods_list, hyp_params_dict
