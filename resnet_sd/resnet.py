
import os

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model


# Helper to build a conv -> BN -> relu block
def conv_bn_relu(input, nb_filter, nb_row, nb_col, W_regularizer, subsample=(1, 1)):
    conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample,
                         kernel_initializer="he_normal", padding="same",kernel_regularizer=W_regularizer,
                                 data_format="channels_first")(input)
                         
    #Conv2D(kernel_size=(3, 3), filters=16, strides=(1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=<keras.reg...)
                         
    norm = BatchNormalization(axis=1)(conv)
    return Activation("relu")(norm)

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def bn_relu_conv(input, nb_filter, nb_row, nb_col, W_regularizer, subsample=(1, 1)):
    norm = BatchNormalization(axis=1)(input)
    activation = Activation("relu")(norm)
    return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample,
                         kernel_initializer="he_normal", padding="same",kernel_regularizer=W_regularizer,
                                 data_format="channels_first")(activation)


# Builds a residual block with repeating bottleneck blocks.
def residual_block(input, block_function, nb_filters, repetations, is_first_layer=False, subsample=False):
    for i in range(repetations):
        init_subsample = (1, 1)
        if i == 0 and (is_first_layer or subsample):
            init_subsample = (2, 2)
        input = block_function(input, nb_filters=nb_filters, init_subsample=init_subsample)
    return input
