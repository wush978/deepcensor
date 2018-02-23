from os.path import isfile, join
from deepcensor.load_data import load_exp_data, wide, deep
from scipy import sparse
import numpy as np

def get_data(config):
    X_wide, y, embed_input_dim, ncol = wide.get_data(config)
    X_deep, y, embed_input_dim, ncol = deep.get_data(config)
    return (X_wide + X_deep, y, embed_input_dim, ncol)
