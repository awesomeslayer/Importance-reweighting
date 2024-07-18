from copy import copy

def errors_init(params, sizes, type, flag):
    err_dict = {}
    if flag:
        for x_temp in params["x"] + params["y"]:
            err_dict[x_temp] = copy(type) 
    else:
        for x_temp in params["x"] + params["x_hyp"] + params["y"]:
                err_dict[x_temp] = copy(type)
    
    err_hyp_dict = {}
    for x_temp in params["x_hyp"]:
        err_hyp_dict[x_temp] = []
        for i in range(sizes[x_temp]):
            err_hyp_dict[x_temp].append(copy(type))

    return err_dict, err_hyp_dict

def find_sizes(params, hyperparams_dict):
    sizes = dict()
    max_size = 0
    for x_temp in params["x_hyp"]:
        sizes[x_temp] = len(hyperparams_dict[x_temp])
        if len(hyperparams_dict[x_temp]) > max_size:
            max_size = len(hyperparams_dict[x_temp])

    sizes["max_size"] = max_size
    return sizes
