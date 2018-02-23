import keras.backend as K
import tensorflow as tf
import numpy as np
from numpy import float32
from ..callback import EpochCoordinateDescent
import scipy

def get_loss_no(X, wp_bp, batch_size = None):
    if batch_size is None:
        batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp)), "ratio" : 1000}
    def training_loss_skip_nan(y_true, y_pred):
        sr = -loglikelihood_skip_nan(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    def loglikelihood_skip_nan(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        y_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        y_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        is_not_nan = tf.logical_not(tf.is_nan(y_true))    
        y_true = tf.boolean_mask(y_true, is_not_nan)
        y_pred = tf.boolean_mask(y_pred, is_not_nan)
        dist = tf.distributions.Normal(loc = y_pred, scale = sigma)
        return dist.log_prob(y_true)
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        wp_pred = y_pred[is_win]
        from scipy.stats import norm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = norm.logpdf(wp, wp_pred, sigma)
            return -(np.sum(wp_loglikelihood))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x[:,0], batch_size = batch_size)
    ]
    return (param, training_loss_skip_nan, loglikelihood_skip_nan, callbacks, lambda x : x, lambda x : x, np.nanmean(wp))



def get_loss_yes(X, wp_bp, batch_size = None):
    if batch_size is None:
        batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp)), "ratio" : 1000}
    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win))
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_rss = tf.square(wp_pred - wp_true)
        wp_loglik = - (wp_rss / (2 * sigma_square) + np.log(2 * np.pi * sigma_square) / 2)
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    def training_loss(y_true, y_pred):
        #.mean(-loglikelihood(y_true, y_pred)) * param["ratio"]
        sr = -loglikelihood(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        from scipy.stats import norm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = norm.logpdf(wp, wp_pred, sigma)
            bp_loglikelihood = norm.logsf(bp, bp_pred, sigma)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    return (param, training_loss, loglikelihood, callbacks, lambda x : x, lambda x : x, np.nanmean(wp))
