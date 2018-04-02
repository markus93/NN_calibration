from __future__ import print_function

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from sklearn.model_selection import train_test_split

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.calibration import evaluate_model

batch_size = 64
nb_classes10 = 10
nb_classes100 = 100

nb_epoch = 300

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation
seed = 333
weight_decay = 0.0001
learning_rate = 0.1

weights_file_10 =  "../../models/weights_densenet_16_8_c10.h5"
weights_file_100 =  "../../models/weights_densenet_16_8_c100.h5"


# Preprocessing for DenseNet https://arxiv.org/pdf/1608.06993v3.pdf
def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test


if __name__ == '__main__':

    # CIFAR-10 ===================

    print("Evaluate CIFAR-10 densenet.")

    model = densenet.DenseNet(img_dim, classes=nb_classes10, depth=depth, nb_dense_block=nb_dense_block,
                              growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, weight_decay=1e-4)

                              
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #For data preprocessing, we normalize the data using the channel means and standard deviations (https://arxiv.org/pdf/1608.06993v3.pdf)
    x_train, x_test = color_preprocessing(x_train, x_test)


    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed


    y_train45 = np_utils.to_categorical(y_train45, nb_classes10)  # 1-hot vector
    y_val = np_utils.to_categorical(y_val, nb_classes10)
    y_test = np_utils.to_categorical(y_test, nb_classes10)
    evaluate_model(model, weights_file_10, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_densenet40_c10", x_val = x_val, y_val = y_val)
