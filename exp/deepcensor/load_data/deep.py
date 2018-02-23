from os.path import isfile, join
from deepcensor.load_data import load_exp_data
from scipy import sparse
import numpy as np

def get_data(config):
    data_root = join("..", config["data_src"], config["data_name"])
    X = load_exp_data(data_root, "lv1")
    y = load_exp_data(data_root, "responses")
    offsets = np.cumsum([X["levels"][key].shape[0] for key, value in X["data"].items()])
    offsets = np.insert(offsets, 0, 0)
    #offsets = np.delete(offsets, offsets.shape[0] - 1)
    offsets = offsets + 1
    embed_input_dim = offsets[offsets.shape[0] - 1]
    offsets = np.delete(offsets, offsets.shape[0] - 1)
    nrow = next (iter (X["data"].values())).shape[0]
    ncol = 0
    for key, value in sorted(X["data"].items()):
        if sparse.issparse(value):
            ncol += np.max(np.diff(value.indptr))
        else:
            ncol += 1
    if isfile(join(data_root, "lv1_embedding.npy")):
        X_all = np.load(join(data_root, "lv1_embedding.npy"))
    else:
        X_all = np.zeros(shape = (nrow, ncol), dtype = "int32")
        for i in range(nrow):
            index = 0
            for offset, (key, value) in zip(offsets, sorted(X["data"].items())):
                if sparse.issparse(value):
                    j_range = range(value.indptr[i], value.indptr[i + 1])
                    X_all[i,range(index, index + len(j_range))] = [value.indices[j] + offset for j in j_range]
                    index += len(j_range)            
                else:
                    X_all[i,index] = value[i] + offset
                    index += 1
            if i % 1000 == 0:
                import sys
                sys.stdout.write('\r')
                sys.stdout.write("%d%%" % (100 / nrow * i))
                sys.stdout.flush()
        X_all = X_all.astype("int32")
        np.save(join(data_root, "lv1_embedding.npy"), X_all)
    return ([X_all], y["data"], embed_input_dim, ncol)
