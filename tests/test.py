from source.estimations import (ISE_clip, importance_sampling_error,
                                importance_sampling_error_degree,
                                monte_carlo_error)
from source.mandoline_estimation import mandoline_error


def test(
    conf,
    hyperparams_params,
    gen_dict,
    target_error,
    hyperparams={},
):
    iter_err = dict()

    if "MCE_p" in target_error:

        iter_err["MCE_p"] = monte_carlo_error(gen_dict["err"], gen_dict["p_test"])

    if "MCE_g" in target_error:

        iter_err["MCE_g"] = monte_carlo_error(gen_dict["err"], gen_dict["g_test"])

    if "ISE_g" in target_error:

        iter_err["ISE_g"] = importance_sampling_error(
            gen_dict["err"], gen_dict["p"], gen_dict["g"], gen_dict["g_test"]
        )

    if "ISE_g_estim" in target_error:
        iter_err["ISE_g_estim"] = importance_sampling_error(
            gen_dict["err"],
            gen_dict["p"],
            gen_dict["g_estim"],
            gen_dict["g_test"],
        )

    if "ISE_g_reg_uniform" in target_error:
        epsilon = hyperparams["ISE_g_reg_uniform"]
        g_estim_new = lambda X: (1 - epsilon) * gen_dict["g_estim"](X) + epsilon / (
            conf["max_mu"] ** 2
        )

        iter_err["ISE_g_reg_uniform"] = importance_sampling_error(
            gen_dict["err"],
            gen_dict["p"],
            g_estim_new,
            gen_dict["g_test"],
        )

    if "ISE_g_reg_degree" in target_error:
        iter_err["ISE_g_reg_degree"] = importance_sampling_error_degree(
            gen_dict["err"],
            gen_dict["p"],
            g_estim_new,
            gen_dict["g_test"],
            hyperparams["ISE_g_reg_degree"],
        )

    if "ISE_g_clip" in target_error:
        iter_err["ISE_g_clip"] = ISE_clip(
            gen_dict["err"],
            gen_dict["p"],
            gen_dict["g"],
            gen_dict["g_test"],
            hyperparams["ISE_g_clip"],
            smooth_flag=hyperparams_params["smooth_flag"],
        )

    if "ISE_g_estim_clip" in target_error:
        iter_err["ISE_g_estim_clip"] = ISE_clip(
            gen_dict["err"],
            gen_dict["p"],
            gen_dict["g_estim"],
            gen_dict["g_test"],
            hyperparams["ISE_g_estim_clip"],
            smooth_flag=hyperparams_params["smooth_flag"],
        )

    if "Mandoline" in target_error:
        iter_err["Mandoline"] = mandoline_error(
            gen_dict,
            n_slices=hyperparams["Mandoline"],
            slice_method=hyperparams_params["slice_method"],
        )

    return iter_err
