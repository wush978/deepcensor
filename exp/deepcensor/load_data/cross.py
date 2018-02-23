from os import rename
from os.path import isfile, join
from deepcensor.load_data import load_exp_data
from scipy import sparse
import numpy as np

def get_data(config):
    data_root = join("..", config["data_src"], config["data_name"])
    y = load_exp_data(data_root, "responses")
    data_path = join(data_root, "linear-X-data.npy")
    indices_path = join(data_root, "linear-X-indices.npy")
    indptr_path = join(data_root, "linear-X-indptr.npy")
    shape_path = join(data_root, "linear-X-shape.npy")
    if isfile(data_path) and isfile(indices_path) and isfile(indptr_path) and isfile(shape_path):
        data = np.load(data_path)
        indices = np.load(indices_path)
        indptr = np.load(indptr_path)
        shape = tuple(np.load(shape_path))
        return ([sparse.csr_matrix((data, indices, indptr), shape = shape)], y["data"], None, None)
    X = load_exp_data(data_root, "lv1")
    for key, value in sorted(X["data"].items()):
        if not sparse.issparse(value):
            indptr = np.array(range(value.shape[0] + 1))
            indices = value
            X["data"][key] = sparse.csr_matrix(
                (np.repeat(1, value.shape), indices, indptr),
                shape = (value.shape[0], X["levels"][key].shape[0])
            )
    X["data"] = sparse.hstack([value for key, value in sorted(X["data"].items())]).tocsr()
    np.save(data_path + ".tmp.npy", X["data"].data)
    rename(data_path + ".tmp.npy", data_path)
    np.save(indices_path + ".tmp.npy", X["data"].indices)
    rename(indices_path + ".tmp.npy", indices_path)
    np.save(indptr_path + ".tmp.npy", X["data"].indptr)
    rename(indptr_path + ".tmp.npy", indptr_path)
    np.save(shape_path + ".tmp.npy", X["data"].shape)
    rename(shape_path + ".tmp.npy", shape_path)
    return ([X["data"]], y["data"], None, None)
