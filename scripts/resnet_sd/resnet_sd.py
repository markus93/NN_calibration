# Code base from https://github.com/transcranial/stochastic-depth/blob/master/cifar10.py

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization, Add
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import np_utils
import keras.backend as K
import json
import time


def _get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    """
    See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    """
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise

            
def _zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def _stochastic_survival(y, p_survival=1.0):
    # binomial random variable
    survival = K.random_binomial((1,), p=p_survival)
    # during testing phase:
    # - scale y (see eq. (6))
    # - p_survival effectively becomes 1 for all layers (no layer dropout)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y, 
                           survival * y)


def _stochastic_depth_residual_block(x, nb_filters=16, block=0, nb_total_blocks=110, subsample_factor=1, weight_decay = 0.0001):
    """
    Stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    
    Residual block consisting of:
    - Conv - BN - ReLU - Conv - BN
    - identity shortcut connection
    - merge Conv path with shortcut path

    Original paper (http://arxiv.org/pdf/1512.03385v1.pdf) then has ReLU,
    but we leave this out: see https://github.com/gcr/torch-residual-networks

    Additional variants explored in http://arxiv.org/pdf/1603.05027v1.pdf
    
    some code adapted from https://github.com/dblN/stochastic_depth_keras
    """
    
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, data_format="channels_last")(x)
        if nb_filters > prev_nb_channels:
            shortcut = Lambda(_zero_pad_channels,
                              arguments={'pad': nb_filters - prev_nb_channels})(shortcut)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    y = Convolution2D(nb_filters, (3, 3), strides=subsample, 
                      padding="same", data_format="channels_last", 
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight_decay))(x)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, (3, 3), strides=(1, 1), 
                      padding="same", data_format="channels_last", 
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight_decay))(y)
    y = BatchNormalization(axis=3)(y)
    
    p_survival = _get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
    y = Lambda(_stochastic_survival, arguments={'p_survival': p_survival})(y)
    
    out = Add()([y, shortcut])

    return out

def resnet_sd_model(img_shape = (32,32), img_channels = 3, layers = 110, nb_classes = 10, verbose = False, weight_decay = 0.0001):

    start_time = time.time()  # Take time

    img_rows, img_cols = img_shape
    blocks_per_group = (layers - 2)//6
    
    if verbose:
        print("Blocks per group:", blocks_per_group)

    inputs = Input(shape=(img_rows, img_cols, img_channels))

    # Create model
    x = Convolution2D(16, (3, 3), padding="same", data_format="channels_last", 
                      kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 1st group
    for i in range(0, blocks_per_group):
        nb_filters = 16
        x = _stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                            block=i, nb_total_blocks=3 * blocks_per_group, 
                                            subsample_factor=1)

    # 2nd group
    for i in range(0, blocks_per_group):
        nb_filters = 32
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = _stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                            block=blocks_per_group + i, nb_total_blocks=3 * blocks_per_group, 
                                            subsample_factor=subsample_factor, weight_decay = weight_decay)
    # 3rd group
    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = _stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                            block=2 * blocks_per_group + i, nb_total_blocks=3 * blocks_per_group, 
                                            subsample_factor=subsample_factor, weight_decay = weight_decay)

    x = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format="channels_last")(x)
    x = Flatten()(x)
#pool_size=(2, 2), 
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)

    if verbose:
        model.summary()
        
    return model
        

# Learning rate schedule
def lr_sch(epoch):
    if epoch < nb_epoch * 0.5:
        return 0.1
    elif epoch < nb_epoch * 0.75:
        return 0.01
    else:
        return 0.001

    
# Main for testing purposes
if __name__ == "__main__":

    batch_size = 128
    nb_epoch = 200    
    nb_classes = 10
    weight_decay = 0.0001

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    mean = np.mean(x_train, axis=0, keepdims=True)
    print(mean.shape)
    std = np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(lr_sch)
    
    model = resnet_sd_model(img_shape = (32,32), img_channels = 3, layers = 110, nb_classes = 10, verbose = True, weight_decay = weight_decay)

    # Model saving callback
    #checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)


    # realtime data augmentation
    datagen_train = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False)
        
    datagen_train.fit(x_train)

    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                                  steps_per_epoch=x_train.shape[0]//batch_size, 
                                  nb_epoch=nb_epoch, verbose=2,
                                  validation_data=(x_test, y_test),
                                  callbacks=[lr_scheduler])
                        
    ## Add evaluation + import

    with open('cifar10_train_history.json', 'w') as f_out:
        json.dump(history.history, f_out)