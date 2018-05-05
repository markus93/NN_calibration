import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import collections
import pickle

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from load_data_svhn import load_data_svhn
from resnet_sd import resnet_sd_model

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.calibration import evaluate_model
    
    
# Per channel mean and std normalization
def color_preprocessing(x_train, x_val, x_test):
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')    
    x_test = x_test.astype('float32')
    
    mean = np.mean(x_train, axis=(0,1,2))  # Per channel mean
    std = np.std(x_train, axis=(0,1,2))
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    
    return x_train, x_test
    

if __name__ == '__main__':

    # constants
    learning_rate = 0.1
    nb_epochs = 50
    batch_size = 128
    nb_classes = 10
    seed = 333
    layers = 152 # n = 25 (152-2)/6
    weights_file = "../../models/resnet_152_SD_SVHN.hdf5"


    # data
    print("Loading data, may take some time and memory!")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data_svhn(seed = seed)
    print(x_train.shape)
    print("Data loaded")
    
    x_train, x_test = color_preprocessing(x_train, x_val, x_test)  # Per channel mean

    
    # Try with ImageDataGenerator, otherwise it takes massive amount of memory
    img_gen = ImageDataGenerator(
        data_format="channels_last"
    )

    img_gen.fit(x_train)


    y_train = np_utils.to_categorical(y_train, nb_classes)  # 1-hot vector
    y_val = np_utils.to_categorical(y_val, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
        
    # building and training net
    model = resnet_sd_model(img_shape = (32,32), img_channels = 3, 
                            layers = layers, nb_classes = nb_classes, verbose = True)

    print("Model compiled")

   evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_resnet152_SD_SVHN", x_val = x_val, y_val = y_val)
    