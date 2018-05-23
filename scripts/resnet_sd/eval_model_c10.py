# Load in model weights and evaluate its goodness (ECE, MCE, error) also saves logits

import numpy as np
import collections
import pickle

from resnet_sd import resnet_sd_model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model


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
    
    return x_train, x_val, x_test        
        
        

if __name__ == '__main__':

    # constants
    img_rows, img_cols = 32, 32
    img_channels = 3
    nb_epochs = 500
    batch_size = 128
    nb_classes = 10
    seed = 333
    weights_file = "../../models/resnet_110_SD_c10.hdf5"


    # data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    
    # Data splitting (get additional 5k validation set)
    # Sklearn to split
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    x_train45, x_val, x_test = color_preprocessing(x_train45, x_val, x_test)  # Mean per channel    

    y_train45 = np_utils.to_categorical(y_train45, nb_classes)  # 1-hot vector
    y_val = np_utils.to_categorical(y_val, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
        
    # building and training net
    model = resnet_sd_model(img_shape = (32,32), img_channels = 3, 
                            layers = 110, nb_classes = nb_classes, verbose = True)
                            
    evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_resnet110_SD_c10", x_val = x_val, y_val = y_val)
    
    

