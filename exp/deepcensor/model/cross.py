from keras.layers import Dense, Activation, Input, Dropout, Layer, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam, Adagrad, Adadelta
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras import initializers, regularizers
from deepcensor.loss import get_loss
from deepcensor.callback import LogMSE, LogVariable
from scipy.sparse import issparse
import tensorflow as tf
import keras.backend as K

class Cross(Layer):
    def __init__(
        self,
        nlayer, 
        kernel_initializer = "glorot_uniform", 
        bias_initializer = "zeros",
        kernel_regularizer = None,
        bias_regularizer = None
    ):
        self.nlayer = nlayer
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        super(Cross, self).__init__()
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.kernel = []
        self.bias = []
        for i in range(self.nlayer):
            self.kernel.append(self.add_weight(
                shape=(input_dim, 1),
                initializer=self.kernel_initializer,
                name='kernel_' + str(i),
                regularizer=self.kernel_regularizer
            ))
            self.bias.append(self.add_weight(
                shape=(1,1),
                initializer=self.bias_initializer,
                name='bias_' + str(i),
                regularizer=self.bias_regularizer
            ))
        self.built = True
    def call(self, x):
        X = [tf.sparse_tensor_to_dense(x, validate_indices=False)]
        for i in range(self.nlayer):
            X_last = X[len(X) - 1]
            r = tf.matmul(X_last, self.kernel[i])
            r = tf.multiply(X[0], r)
            X.append(r + self.bias[i] + X_last)
        return X[len(X) - 1]
    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape

# setting loss, callback and parameters according to config["loss"]
def get_model(config, training_loss, X_train, bias_init, embed_input_dim, ncol):
    input = Input(shape = (X_train[0].shape[1],), name = "input", sparse = True)
    cross_layer = Cross(config["nlayer"])(input)
    wp_output = Dense(
        1, W_regularizer = l2(config["l2"]), activation = "linear", name = "wp_output",
        kernel_initializer = "zeros",
        bias_initializer = Constant(value=bias_init)
    )(cross_layer)
    wp_output = Concatenate()([wp_output, wp_output])
    model = Model(input = [input], output = [wp_output])
    model.compile(optimizer = Adadelta(), loss = training_loss)
    return model
