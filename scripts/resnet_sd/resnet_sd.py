import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization
from keras.engine import merge, Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time

nb_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32'), (0, 2, 3, 1))
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = np.transpose(x_test.astype('float32'), (0, 2, 3, 1))
x_test = (x_test - mean) / std
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    """
    See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    """
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise

            
def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def stochastic_survival(y, p_survival=1.0):
    # binomial random variable
    survival = K.random_binomial((1,), p=p_survival)
    # during testing phase:
    # - scale y (see eq. (6))
    # - p_survival effectively becomes 1 for all layers (no layer dropout)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y, 
                           survival * y)


def stochastic_depth_residual_block(x, nb_filters=16, block=0, nb_total_blocks=110, subsample_factor=1):
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
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
        if nb_filters > prev_nb_channels:
            shortcut = Lambda(zero_pad_channels,
                              arguments={'pad': nb_filters - prev_nb_channels})(shortcut)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)
    
    p_survival = get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
    y = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(y)
    
    out = merge([y, shortcut], mode='sum')

    return out


start_time = time.time()

img_rows, img_cols = 32, 32
img_channels = 3

blocks_per_group = 33

inputs = Input(shape=(img_rows, img_cols, img_channels))

x = Convolution2D(16, 3, 3, 
                  init='he_normal', border_mode='same', dim_ordering='tf')(inputs)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

for i in range(0, blocks_per_group):
    nb_filters = 16
    x = stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                        block=i, nb_total_blocks=3 * blocks_per_group, 
                                        subsample_factor=1)

for i in range(0, blocks_per_group):
    nb_filters = 32
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                        block=blocks_per_group + i, nb_total_blocks=3 * blocks_per_group, 
                                        subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 64
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                        block=2 * blocks_per_group + i, nb_total_blocks=3 * blocks_per_group, 
                                        subsample_factor=subsample_factor)

x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
x = Flatten()(x)

predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)

model.summary()

print('model init time: {}'.format(time.time() - start_time))

start_time = time.time()

sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('model compile time: {}'.format(time.time() - start_time))

batch_size = 128
nb_epoch = 200
data_augmentation = True

# Learning rate schedule
def lr_sch(epoch):
    if epoch < nb_epoch * 0.5:
        return 0.1
    elif epoch < nb_epoch * 0.75:
        return 0.01
    else:
        return 0.001

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_sch)

# Model saving callback
#checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                        validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[lr_scheduler])
else:
    print('Using real-time data augmentation.')

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
                                  samples_per_epoch=x_train.shape[0], 
                                  nb_epoch=nb_epoch, verbose=2,
                                  validation_data=(x_test, y_test),
                                  callbacks=[lr_scheduler])

with open('cifar10_train_history.json', 'w') as f_out:
    json.dump(history.history, f_out)