from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization, Embedding, Concatenate, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam, Adagrad, Adadelta
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from deepcensor.loss import get_loss
from deepcensor.callback import LogMSE, LogVariable
from scipy.sparse import issparse
import tensorflow as tf

# setting loss, callback and parameters according to config["loss"]
def get_model(config, training_loss, X_train, bias_init, embed_input_dim, ncol):
    input = Input(shape = (ncol,), name = "input")
    embedding_layer = Embedding(embed_input_dim, 6, input_length = ncol)(input)
    flatten = Flatten()(embedding_layer)
    layer = flatten
    for i in range(config["nlayer"]):
        layer = Dense(config["nnode"], activation = "relu")(layer)
    wp_output = Dense(
        1, W_regularizer = l2(config["l2"]), activation = "linear", name = "wp_output",
        kernel_initializer = "uniform",
        bias_initializer = Constant(value=bias_init)
    )(layer)
    wp_output = Concatenate()([wp_output, wp_output])
    model = Model(input = [input], output = [wp_output])
    model.compile(optimizer = Adadelta(), loss = training_loss)
    return model
