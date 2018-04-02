# -*- coding: utf-8 -*-

import numpy as np
   
from keras.optimizers import SGD
from densenet161 import DenseNet 
from load_data_imagenet import load_data_imagenet_split
from resnet152 import resnet152_model
import keras

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.calibration import evaluate_model

# Constants
MEAN = [103.939, 116.779, 123.68]

if __name__ == '__main__':

    print("Evaluate ResNet152")

    ## Load already split and resized data (mean subtracted)
    seed = 333
    num_classes = 1000
    weights_file_resnet = "../../models/resnet152_weights_tf.h5"
    weights_file_densenet = "../../models/densenet161_weights_tf.h5"


    (x_val, y_val), (x_test, y_test) = load_data_imagenet_split(seed = seed)
    
    print("Data loaded.")

    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("Preprocess data.")
    
    x_test = x_test[..., ::-1]
    x_val = x_val[..., ::-1]
    
    for i in range(3):
        x_test[:,:,:,i] -= MEAN[i]
        x_val[:,:,:,i] -= MEAN[i]

    model = resnet152_model()
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Start evaluation!")
    evaluate_model(model, weights_file_resnet, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_resnet152_imgnet", x_val = x_val, y_val = y_val)
    

    print("Evaluate DenseNet161")
    
    model = DenseNet(reduction=0.5, classes=1000)

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Subtract mean pixel and multiple by scaling constant # pixel mean subtracted previously
    # Reference: https://github.com/shicai/DenseNet-Caffe
    
    for i in range(3):
        x_test[:,:,:,i] = x_test[:,:,:,i] * 0.017
        x_val[:,:,:,i] = x_val[:,:,:,i] * 0.017
    
    print("Evaluation for second model.")
    evaluate_model(model, weights_file_densenet, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_densenet161_imgnet", x_val = x_val, y_val = y_val)

