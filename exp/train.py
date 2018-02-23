import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import argparse
import json
import sys

# loading configuration
if hasattr(sys, "ps1"):
    # interactive
    with open("linear-normal-no-ipinyou.exp.data-201310_1e-4/01.json") as f:
        config = json.load(f)
    epoch = 2
else:
    parser = argparse.ArgumentParser(description = "LR exp program")
    parser.add_argument('--config', help = 'configuration file of the experiments')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    if "DEBUG" in os.environ:
        epoch = 2
    else:
        epoch = 1000

from deepcensor.load_data import get_data, get_simulated_bidding
from deepcensor.loss import get_loss
from deepcensor.callback import LogMSE, LogVariable, NanStopping
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
from deepcensor.model import get_model
import copy
import tensorflow as tf
from os.path import splitext

X, y, embed_input_dim, ncol = get_data(config)
# get simulated bidding results
y_sim = get_simulated_bidding(y)

data = [y_sim["bp"],y_sim["clk"],y_sim["is_win"],y_sim["wp"],y["clk"],y["wp"]] + X
sim_bp_index = 0
sim_clk_index = 1
sim_is_win_index = 2
sim_wp_index = 3
clk_index = 4
wp_index = 5

# split train / valid / test(fixed) dataset
data_train_test = train_test_split(*data, test_size = 0.1, random_state = config["random_state_test"])
data_train = data_train_test[0::2]
data_test = data_train_test[1::2]
data_train_valid = train_test_split(*data_train, test_size = 0.1, random_state = config["random_state_valid"])
data_train = data_train_valid[0::2]
data_valid = data_train_valid[1::2]

wp_bp_train, wp_bp_valid, wp_bp_test = [
    np.column_stack((data[sim_wp_index], data[sim_bp_index])) 
    for data in [data_train, data_valid, data_test]
]

allwp_bp_train, allwp_bp_valid, allwp_bp_test = [
    np.column_stack((data[wp_index], data[sim_bp_index])) 
    for data in [data_train, data_valid, data_test]
]

def get_loose_wp_bp(data):
    wp_bp = np.column_stack((data[wp_index], data[sim_bp_index]))
    wp_bp[data[sim_is_win_index],:] = np.nan
    return wp_bp
losewp_bp_train, losewp_bp_valid, losewp_bp_test = [
    get_loose_wp_bp(data)
    for data in [data_train, data_valid, data_test]
]

X_train = data_train[(wp_index + 1):]
X_valid = data_valid[(wp_index + 1):]
X_test = data_test[(wp_index + 1):]

# setting loss, callback and parameters according to config["loss"]
param, training_loss, evaluate_loglikelihood, callbacks, mse_pred, mae_pred, bias_init = get_loss(config, X_train, wp_bp_train, batch_size = 4096)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_session = tf.Session(config=tf_config)
import keras
keras.backend.set_session(tf_session)

model = get_model(config, training_loss, X_train, bias_init, embed_input_dim, ncol)
history = model.fit(
    X_train,
    wp_bp_train, 
    epochs = epoch, batch_size = 1024, verbose = 1, 
    shuffle = True, validation_data = (X_valid, wp_bp_valid),
    callbacks = callbacks + [
        LogVariable(config["loss"] + "-" + config["censoring"] + "-param", param),
        LogMSE(["mse", "allmse", "losemse"], X_train, [wp_bp_train, allwp_bp_train, losewp_bp_train], mse_pred, output_preprocessor = lambda x : x[:,0], sparse = True, batch_size = 4096),
        LogMSE(["val_mse", "val_allmse", "val_losemse"], X_valid, [wp_bp_valid, allwp_bp_valid, losewp_bp_valid], mse_pred, output_preprocessor = lambda x : x[:,0], sparse = True, batch_size = 4096),
        NanStopping(),
        EarlyStopping(monitor = "val_loss", min_delta = 1e-6, patience = 5, mode = "min")
    ]
)

# evaluation
p_train = model.predict(X_train, batch_size = 4096)
p_valid = model.predict(X_valid, batch_size = 4096)
p_test = model.predict(X_test, batch_size = 4096)

# prepare output
check = copy.deepcopy(config)

def check_wp(data):
    return {
        "winning_mean" : np.nanmean(data[sim_wp_index]).tolist(),
        "losing_mean" : np.nanmean(data[wp_index][np.invert(data[sim_is_win_index])]).tolist(),
        "all_mean" : np.nanmean(data[wp_index]).tolist(),
        "winning_std" : np.nanstd(data[sim_wp_index]).tolist(),
        "losing_std" : np.nanstd(data[wp_index][np.invert(data[sim_is_win_index])]).tolist(),
        "all_std" : np.nanstd(data[wp_index]).tolist()
    }
check["wp_train"] = check_wp(data_train)
check["wp_valid"] = check_wp(data_valid)
check["wp_test"] = check_wp(data_test)

# calculate losses

def get_mean_sd(x):
    return {"mean" : np.nanmean(x).tolist(), "std" : np.nanstd(x).tolist() / np.count_nonzero(~np.isnan(x))}

def get_winning_likelihood(data, p):
    wp_bp = np.column_stack(
        (data[sim_wp_index], data[sim_bp_index])
    )
    sess = tf.keras.backend.get_session()
    return sess.run(evaluate_loglikelihood(tf.convert_to_tensor(wp_bp, dtype = "float32"), tf.convert_to_tensor(p))).tolist()

def get_losing_likelihood(data, p):
    wp_bp = np.column_stack(
        (data[wp_index][np.invert(data[sim_is_win_index])], data[sim_bp_index][np.invert(data[sim_is_win_index])])
    )
    p = p[np.invert(data[sim_is_win_index]),]
    sess = tf.keras.backend.get_session()
    return sess.run(evaluate_loglikelihood(tf.convert_to_tensor(wp_bp, dtype = "float32"), tf.convert_to_tensor(p))).tolist()

def get_all_likelihood(data, p):
    wp_bp = np.column_stack(
        (data[wp_index], data[wp_index])
    )
    sess = tf.keras.backend.get_session()
    return sess.run(evaluate_loglikelihood(tf.convert_to_tensor(wp_bp, dtype = "float32"), tf.convert_to_tensor(p))).tolist()

def get_winning_mse(data, p):
    return np.square(data[sim_wp_index] - mse_pred(p[:,0]))

def get_losing_mse(data, p):
    return np.square(data[wp_index][np.invert(data[sim_is_win_index])] - mse_pred(p[np.invert(data[sim_is_win_index]),0]))

def get_all_mse(data, p):
    return np.square(data[wp_index] - mse_pred(p[:,0]))

def get_winning_mae(data, p):
    return np.abs(data[sim_wp_index] - mae_pred(p[:,0]))

def get_losing_mae(data, p):
    return np.abs(data[wp_index][np.invert(data[sim_is_win_index])] - mae_pred(p[np.invert(data[sim_is_win_index]),0]))

def get_all_mae(data, p):
    return np.abs(data[wp_index] - mae_pred(p[:,0]))

censoring_type = ["winning", "losing", "all"]
for data_type in ["train", "valid", "test"]:
    check[data_type] = {}
    for measure_type in (
        [ctype + "_" + "likelihood" for ctype in censoring_type] +
        [ctype + "_" + "mse" for ctype in censoring_type] +
        [ctype + "_" + "mae" for ctype in censoring_type]
    ):
        check[data_type][measure_type] = get_mean_sd(globals()["get_" + measure_type](globals()["data_" + data_type], globals()["p_" + data_type]))
# output

if "DEBUG" in os.environ:
    model.save("/tmp/debug.h5")
    with open("/tmp/debug.history", "w") as fp:
        json.dump(history.history, fp)
    with open(splitext("/tmp/debug.history")[0] + ".check", 'w') as fp:
        json.dump(check, fp) 
else:
    model.save(config["output"]["model"])
    with open(config["output"]["history"], 'w') as fp:
        json.dump(history.history, fp)
    with open(splitext(config["output"]["history"])[0] + ".check", 'w') as fp:
        json.dump(check, fp)
