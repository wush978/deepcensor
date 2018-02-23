import keras.backend as K
import tensorflow as tf
import numpy as np
from numpy import float32
from ..callback import EpochCoordinateDescent
import scipy

EULER_MASCH = 0.577215664901532
def get_mse_pred(param):
    def mse_pred(x):
        sigma = K.get_value(param["sigma"])
        return x + EULER_MASCH * sigma
    return mse_pred

LOG_LOG_2 = np.log(np.log(2))
def get_mae_pred(param):
    def mae_pred(x):
        sigma = K.get_value(param["sigma"])
        return x - LOG_LOG_2 * sigma
    return mae_pred


def get_loss_no(X, wp_bp, batch_size = None):
    if batch_size is None:
        batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp) * np.sqrt(6) / np.pi), "ratio" : 1000}
    def training_loss_skip_nan(y_true, y_pred):
        sr = -loglikelihood_skip_nan(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    def loglikelihood_skip_nan(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        log_sigma = np.log(sigma)
        y_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        y_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        is_not_nan = tf.logical_not(tf.is_nan(y_true))    
        y_true = tf.boolean_mask(y_true, is_not_nan)
        y_pred = tf.boolean_mask(y_pred, is_not_nan)
        mu = y_pred
        z = (y_true - mu) / sigma
        return - (z + tf.exp(-z) ) - log_sigma
    def param_updater(y_true, y_pred, param, lower = None):
        #print(K.get_value(param["sigma"]))
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        wp_pred = y_pred[is_win,0]
        wp_residual = wp - wp_pred
        def sigma_loglikelihood(sigma):
            z = wp_residual / sigma
            wp_loglikelihood = - (z + np.exp(-z)) - np.log(sigma)
            return -(np.sum(wp_loglikelihood))
        from scipy.optimize import fmin_l_bfgs_b
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    return (param, training_loss_skip_nan, loglikelihood_skip_nan, callbacks, get_mse_pred(param), get_mae_pred(param), np.nanmean(wp) - EULER_MASCH * np.nanstd(wp) * np.sqrt(6) / np.pi)

def get_loss_yes(X, wp_bp, batch_size = None):
    if batch_size is None:
        batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp) * np.sqrt(6) / np.pi), "ratio" : 1000}
    def training_loss(y_true, y_pred):
        sr = -loglikelihood(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        log_sigma = np.log(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        a = tf.exp(- (bp - wp_pred) / sigma)
        bp_loglik1 = tf.log(-tf.expm1(-a))
        bp_loglik2 = tf.log1p(-tf.exp(-a))
        bp_loglik = tf.where(a < 0.693, bp_loglik1, bp_loglik2)
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win))
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        z = (wp_true - wp_pred) / sigma
        wp_loglik = - (z + tf.exp(-z) ) - log_sigma
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        return tf.where(is_win, wp_loglik, bp_loglik)
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        wp_residual = wp - wp_pred
        bp_pred = y_pred[~is_win,0]
        bp_residual = bp - bp_pred
        def sigma_loglikelihood(sigma):
            #bp_loglikelihood = np.log(1 - np.exp(-np.exp(- bp_residual / sigma)))
            a = np.exp(- bp_residual / sigma)
            is_1 = a < 0.693
            bp_loglikelihood1 = np.log(-np.expm1(-a[is_1]))
            bp_loglikelihood2 = np.log1p(-np.exp(-a[~is_1]))
            z = wp_residual / sigma
            wp_loglikelihood = - (z + np.exp(-z)) - np.log(sigma)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood1) + np.sum(bp_loglikelihood2))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    return (param, training_loss, loglikelihood, callbacks, get_mse_pred(param), get_mae_pred(param), np.nanmean(wp) - EULER_MASCH * np.nanstd(wp) * np.sqrt(6) / np.pi)
