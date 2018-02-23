import numpy as np
from sklearn import metrics
from scipy.stats import linregress
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
        
class LogAUC(Callback):
    def __init__(self, name, X, y, sparse = False, print_every_n = 1, output_index = 0, batch_size = 1024):
        self.name = name
        self.X = X
        self.y = y
        self.print_every_n = print_every_n
        self.output_index = output_index
        self.batch_size = batch_size
    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        if self.name not in self.params['metrics']:
            self.params['metrics'].append(self.name)
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % self.print_every_n == 0:
            logs = logs or {}
            logs[self.name] = 0
            pred = self.model.predict(self.X, verbose = 0, batch_size = self.batch_size)
            y_true = self.y
            y_pred = pred
            if type(y_pred) is list:
                y_pred = y_pred[self.output_index]
            index = np.isnan(y_true) == False
            auc = metrics.roc_auc_score(y_true[index], y_pred[index])
            logs[self.name] = auc

class LogMSE(Callback):
    def __init__(self, name_list, X, y_list, mse_pred, output_preprocessor = lambda x : x, sparse = False, print_every_n = 1, output_index = 0, batch_size = 1024):
        if type(name_list) is list:
            self.name_list = name_list
        else:
            self.name_list = [name_list]
        self.X = X
        if type(y_list) is list:
            self.y_list = y_list
        else:
            self.y_list = [y_list]
        self.print_every_n = print_every_n
        self.output_index = output_index
        self.batch_size = batch_size
        self.mse_pred = mse_pred
        self.output_preprocessor = output_preprocessor
    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        for name in self.name_list:
            if name not in self.params['metrics']:
                self.params['metrics'].append(name)
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % self.print_every_n == 0:
            logs = logs or {}
            pred = self.model.predict(self.X, verbose = 0, batch_size = self.batch_size)
            for name, y in zip(self.name_list, self.y_list):
                logs[name] = 0
                y_true = y
                y_pred = pred
                if type(y_pred) is list:
                    y_pred = y_pred[self.output_index]
                index = np.isnan(y_true) == False
                mse = np.nanmean(np.square(self.output_preprocessor(y_true) - self.mse_pred(self.output_preprocessor(y_pred))))
                logs[name] = mse

class EpochCoordinateDescent(Callback):
    def __init__(self, param, X, y, param_updater, input_output_preprocessor = lambda x : x, update_every_n = 1, batch_size = 1024):
        self.param = param
        self.X = X
        self.y = input_output_preprocessor(y)
        self.param_updater = param_updater
        self.update_every_n = update_every_n
        self.batch_size = batch_size
        self.input_output_preprocessor = input_output_preprocessor
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % self.update_every_n == 0:
            pred = self.model.predict(self.X, batch_size = self.batch_size)
            pred = self.input_output_preprocessor(pred)
            self.param_updater(self.y, pred, self.param)

class LogVariable(Callback):
    def __init__(self, name, param):
        self.param = {}
        for key, value in param.items():
            if type(value) is tf.Variable:
                self.param[name + "_" + key] = value
    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        for key, value in self.param.items():
            if key not in self.params["metrics"]:
                self.params["metrics"].append(key)
    def on_epoch_end(self, epoch, logs = {}):
        for key, value in self.param.items():
            logs[key] = K.get_value(value).tolist()

class NanStopping(Callback):
    def __init__(self):
        pass
    def on_epoch_end(self, epoch, logs = {}):
        if np.isnan(logs.get("loss")):
            self.model.stop_training = True



