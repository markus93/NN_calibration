# Loads in weights and evaluates model (ECE, MCE, error rate) and saves logits.

import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

batch_size    = 128
epochs        = 200
iterations    = 45000 // batch_size
num_classes10   = 10
num_classes100 = 100
weight_decay  = 0.0001
seed = 333
N = 1

weights_file_10 = "../../models/lenet_5_c10.h5"
weights_file_100 = "../../models/lenet_5_c100.h5"

def build_model(n=1, num_classes = 10):
    """
    parameters:
        n: (int) scaling for model (n times filters in Conv2D and nodes in Dense)
    """
    model = Sequential()
    model.add(Conv2D(n*6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(n*16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(n*120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(n*84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

    
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

    print("Evaluate CIFAR-10 - LeNet 5")
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = color_preprocessing(x_train, x_test)
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed

    y_train45 = keras.utils.to_categorical(y_train45, num_classes10)
    y_val = keras.utils.to_categorical(y_val, num_classes10)
    y_test = keras.utils.to_categorical(y_test, num_classes10)


    # build network
    model = build_model(n = N, num_classes = num_classes10)
    evaluate_model(model, weights_file_10, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_lenet5_c10", x_val = x_val, y_val = y_val)
