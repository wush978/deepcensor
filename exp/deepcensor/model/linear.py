from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization, Embedding, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam, Adagrad, Adadelta
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from deepcensor.loss import get_loss
from deepcensor.callback import LogMSE, LogVariable
from scipy.sparse import issparse

def get_model(config, training_loss, X_train, bias_init, embed_input_dim, ncol):
    input = Input(shape = (X_train[0].shape[1],), name = "input", sparse = True)
    wp_output = Dense(
        1, W_regularizer = l2(config["l2"]), activation = "linear", name = "wp_output",
        kernel_initializer = "zeros",
        bias_initializer = Constant(value=bias_init)
    )(input)
    wp_output = Concatenate()([wp_output, wp_output])
    model = Model(input = [input], output = [wp_output])
    model.compile(optimizer = Adadelta(), loss = training_loss)
    return model
