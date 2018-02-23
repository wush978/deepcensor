import h5py
import numpy as np
import os
from os.path import isfile, join
from scipy import sparse

def load_exp_data(data_root, group_name):
    data_by_date = sorted([int(f) for f in os.listdir(data_root) if not isfile(join(data_root, f))])
    result = {}
    result_levels = {}
    for f in data_by_date:
        with h5py.File(join(data_root, str(f), "exp.data.h5"), mode = "r+") as h5f:
            for variable, group in h5f[group_name].items():
                if not variable in result.keys():
                    result[variable] = []
                value = group[".value"][:]
                value_class = group["class"][0].decode("utf-8")
                if "levels" in group.keys():
                    levels = group["levels"][:]
                else:
                    levels = None
                if not variable in result_levels.keys():
                    result_levels[variable] = levels
                is_na = value.dtype == np.dtype("uint8") and 2 in value
                if not is_na:
                    # dtype("uint8") is NA_int in R
                    if value_class == "integer":
                        result[variable].append(value)
                    elif value_class == "numeric":
                        result[variable].append(value)
                    elif value_class == "logical":
                        result[variable].append(value)
                    elif value_class == "compressed.list":
                        element_size = group["element.size"][:]
                        indptr = np.insert(np.cumsum(element_size), 0, 0)
                        indices = value
                        value = sparse.csr_matrix(
                            (np.repeat(1, indices.shape), indices, indptr), 
                            shape = (indptr.shape[0] - 1, levels.shape[0])
                        )
                        result[variable].append(value)
                    else:
                        raise RuntimeError("unknown value_class: " + value_class)
    for key, value in result.items():
        if type(value[0]) is np.ndarray:
            result[key] = np.concatenate(value)
        elif sparse.issparse(value[0]):
            result[key] = sparse.vstack(value)
        else:
            raise RuntimeError("Unknown type of element: " + key)
    return {"data" : result, "levels" : result_levels}

# for generating simulated bidding results
def get_simulated_bidding(y, ratio = 0.5):
    bp = np.copy(y["bp"]) * ratio
    is_win = bp > y["wp"]
    wp = np.copy(y["wp"])
    wp[np.invert(is_win)] = np.nan
    clk = np.copy(y["clk"])
    clk[np.invert(is_win)] = -1
    return {"bp":bp,"wp":wp,"clk":clk,"is_win":is_win}

import importlib
def get_data(config):
    module = importlib.import_module('deepcensor.load_data.' + config["structure"])
    return module.get_data(config)
