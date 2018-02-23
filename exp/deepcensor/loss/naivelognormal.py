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
    param = {"sigma" : K.variable(np.nanstd(np.log(wp))), "ratio" : 1000}
    def training_loss_skip_nan(y_true, y_pred):
        y_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        y_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        is_not_nan = tf.logical_not(tf.is_nan(y_true))    
        y_true = tf.boolean_mask(y_true, is_not_nan)
        y_pred = tf.boolean_mask(y_pred, is_not_nan)
        y_true = tf.log(y_true)
        return K.mean(tf.square(y_true - y_pred))
    def loglikelihood_skip_nan(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = sigma * sigma
        y_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        y_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        is_not_nan = tf.logical_not(tf.is_nan(y_true))    
        y_true = tf.boolean_mask(y_true, is_not_nan)
        y_pred = tf.boolean_mask(y_pred, is_not_nan)
        ds = tf.contrib.distributions
        log_normal = ds.TransformedDistribution(
          distribution=ds.Normal(loc=y_pred, scale=sigma),
          bijector=ds.bijectors.Exp())
        return log_normal.log_prob(y_true)
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        wp_pred = y_pred[is_win]
        wp_pred_exp = np.exp(wp_pred)
        from scipy.stats import lognorm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = lognorm.logpdf(wp, s = sigma, scale = wp_pred_exp)
            return -(np.sum(wp_loglikelihood))
        from scipy.optimize import fmin_l_bfgs_b
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x[:,0], batch_size = batch_size)
    ]
    def mse_pred(x):
        sigma = K.get_value(param["sigma"])
        return np.exp(x + sigma * sigma / 2)
    def mae_pred(x):
        return np.exp(x)
    return (param, training_loss_skip_nan, loglikelihood_skip_nan, callbacks, mse_pred, mae_pred, np.nanmean(np.log(wp)))



def get_loss_yes(X, wp_bp, batch_size = None):
    if batch_size is None:
        batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(np.log(wp))), "ratio" : 1000}
    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        y_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        ds = tf.contrib.distributions
        dist = ds.TransformedDistribution(
          distribution=ds.Normal(loc=y_pred, scale=sigma),
          bijector=ds.bijectors.Exp())
#        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win))
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(y_pred, is_win)
        wp_dist = ds.TransformedDistribution(
          distribution=ds.Normal(loc=wp_pred, scale=sigma),
          bijector=ds.bijectors.Exp())
        wp_loglik = wp_dist.log_prob(wp_true)
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    def training_loss(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.log(tf.unstack(y_true, num = 2, axis = 1)[0])
        bp = tf.log(tf.unstack(y_true, num = 2, axis = 1)[1])
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
        return K.mean(-loglik) * param["ratio"]
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        wp_pred_exp = np.exp(wp_pred)
        bp_pred_exp = np.exp(bp_pred)
        from scipy.stats import lognorm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = lognorm.logpdf(wp, s = sigma, scale = wp_pred_exp)
            bp_loglikelihood = lognorm.logsf(bp, s = sigma, scale = bp_pred_exp)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    def mse_pred(x):
        sigma = K.get_value(param["sigma"])
        return np.exp(x + sigma * sigma / 2)
    def mae_pred(x):
        return np.exp(x)
    return (param, training_loss, loglikelihood, callbacks, mse_pred, mae_pred, np.nanmean(np.log(wp)))
