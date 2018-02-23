from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization, Embedding, Concatenate, Flatten, add
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
    input_wide = Input(shape = (X_train[0].shape[1],), name = "input_wide", sparse = True)
    merge_layer_wide = Dense(
        1, W_regularizer = l2(config["l2"]), activation = "linear", name = "wide_output",
        kernel_initializer = "zeros",
        bias_initializer = Constant(value = bias_init)
    )(input_wide)
    # deep
    input_deep = Input(shape = (ncol,), name = "input_deep")
    embedding_layer = Embedding(embed_input_dim, 6, input_length = ncol)(input_deep)
    flatten = Flatten()(embedding_layer)
    layer = flatten
    for i in range(config["nlayer"]):
        layer = Dense(config["nnode"], activation = "relu")(layer)
    merge_layer_deep = Dense(
        1, W_regularizer = l2(config["l2"]), activation = "linear", name = "deep_output",
        kernel_initializer = "uniform",
        use_bias = False
    )(layer)
    wp_output = add([merge_layer_wide, merge_layer_deep])
    wp_output = Concatenate()([wp_output, wp_output])
    model = Model(input = [input_wide, input_deep], output = [wp_output])
    model.compile(optimizer = Adadelta(), loss = training_loss)
    return model
