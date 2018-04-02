# Used code from https://github.com/preddy5/Residual-Learning-and-Stochastic-Depth as base
# coding: utf-8

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=None'
import numpy as np
import collections
import pickle

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras.callbacks import (
    Callback,
    LearningRateScheduler,
)
from keras.layers import (
    Input,
    Activation,
    Add,
    Dense,
    Flatten,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from resnet import (
    bn_relu_conv,
    conv_bn_relu,
    residual_block
)

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.calibration import evaluate_model

# constants
learning_rate = 0.1
momentum = 0.9
img_rows, img_cols = 32, 32
img_channels = 3
nb_epochs = 500
batch_size = 128
nb_classes = 10
nb_classes100 = 100
pL = 1  # For testing mode
weight_decay = 1e-4
seed = 333
weights_file_10 = "../../models/model_weight_ep500_110SD_cifar_10.hdf5"
weights_file_100 = "../../models/model_weight_ep500_110SD_cifar_100.hdf5"


def _bottleneck(input, nb_filters, init_subsample=(1, 1)):
    conv_1_1 = bn_relu_conv(input, nb_filters, 3, 3, W_regularizer=l2(weight_decay), subsample=init_subsample)
    conv_3_3 = bn_relu_conv(conv_1_1, nb_filters, 3, 3, W_regularizer=l2(weight_decay))
    return _shortcut(input, conv_3_3)

    
def _shortcut(input, residual):
    stride_width = input._keras_shape[2] // residual._keras_shape[2]
    stride_height = input._keras_shape[3] // residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual._keras_shape[1], kernel_size=(1, 1), strides=(stride_width, stride_height),
                         kernel_initializer="he_normal", padding="valid",kernel_regularizer=l2(weight_decay),
                                 data_format="channels_first")(input)
        shortcut = Activation("relu")(shortcut)

    M1 = Add()([shortcut, residual])
    M1 = Activation("relu")(M1)
    
    gate = K.variable(0, dtype="int16")
    decay_rate = 1
    name = 'residual_'+str(len(gates)+1)
    gates[name]=[decay_rate, gate]
    return Lambda(lambda outputs: K.switch(gate, outputs[0], outputs[1]),
                  output_shape= lambda x: x[0], name=name)([shortcut, M1])


# http://arxiv.org/pdf/1512.03385v1.pdf
# 110 Layer resnet
    # repetations: = n*6 + 2 = 18*6 + 2 = 110
    # nr_classes: int - how many classes in the end
def resnet(n = 18, nr_classes=10):
    input = Input(shape=(img_channels, img_rows, img_cols))

    conv1 = conv_bn_relu(input, nb_filter=16, nb_row=3, nb_col=3, W_regularizer=l2(weight_decay))  # Filters, filter_size

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = residual_block(conv1, block_fn, nb_filters=16, repetations=n, is_first_layer=True)
    block2 = residual_block(block1, block_fn, nb_filters=32, repetations=n)
    block3 = residual_block(block2, block_fn, nb_filters=64, repetations=n, subsample=True)
    
    # Classifier block
    pool2 = AveragePooling2D(pool_size=(8, 8))(block3)
    flatten1 = Flatten()(pool2)
    final = Dense(units=nr_classes, kernel_initializer="he_normal", activation="softmax", kernel_regularizer=l2(weight_decay))(flatten1)

    model = Model(inputs=input, outputs=final)
    return model



def set_decay_rate():
    for index, key in enumerate(gates):
        gates[key][0] = 1.0 - float(index)*pL / len(gates)

# Callbacks for updating gates and learning rate
def scheduler(epoch):

    if epoch < 2:
        return learning_rate*0.1
    elif epoch < nb_epochs/2:
        return learning_rate
    elif epoch < nb_epochs*3/4:
        return learning_rate*0.1
    return learning_rate*0.01


class Gates_Callback(Callback):
    def on_batch_begin(self, batch, logs={}):
        probs = np.random.uniform(size=len(gates))
        for i,j in zip(gates, probs):
            if j > gates[i][0]:
                K.set_value(gates[i][1], 1)
            else:
                K.set_value(gates[i][1], 0)

    def on_train_end(self, logs={}):
        for i in gates:
            K.set_value(gates[i][1],1)
            
            

# CIFAR-10 ============================================
print("Evaluation CIFAR-10 SD")

# TODO data loader script?
# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.transpose(x_train.astype('float32'), (0, 3, 1, 2))  # Channels first
x_test = np.transpose(x_test.astype('float32'), (0, 3, 1, 2))  # Channels first


# Data splitting (get additional 5k validation set)
# Sklearn to split
x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed

img_mean = x_train45.mean(axis=0)  # per-pixel mean
img_std = x_train45.std(axis=0)
x_train45 = (x_train45-img_mean)/img_std
x_val = (x_val-img_mean)/img_std
x_test = (x_test-img_mean)/img_std


y_train45 = np_utils.to_categorical(y_train45, nb_classes)  # 1-hot vector
y_val = np_utils.to_categorical(y_val, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


# building and training net
gates=collections.OrderedDict()
model = resnet(nr_classes=nb_classes)


evaluate_model(model, weights_file_10, x_test, y_test, bins = 15, verbose = True)

# CIFAR-100 ======================================
print("Evaluation CIFAR-100 SD")

# TODO data loader script?
# data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = np.transpose(x_train.astype('float32'), (0, 3, 1, 2))  # Channels first
x_test = np.transpose(x_test.astype('float32'), (0, 3, 1, 2))  # Channels first


# Data splitting (get additional 5k validation set)
# Sklearn to split
x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed

img_mean = x_train45.mean(axis=0)  # per-pixel mean
img_std = x_train45.std(axis=0)
x_train45 = (x_train45-img_mean)/img_std
x_val = (x_val-img_mean)/img_std
x_test = (x_test-img_mean)/img_std


y_train45 = np_utils.to_categorical(y_train45, nb_classes100)  # 1-hot vector
y_val = np_utils.to_categorical(y_val, nb_classes100)
y_test = np_utils.to_categorical(y_test, nb_classes100)

gates=collections.OrderedDict()
model = resnet(nr_classes=nb_classes100)
evaluate_model(model, weights_file_100, x_test, y_test, bins = 15, verbose = True)

