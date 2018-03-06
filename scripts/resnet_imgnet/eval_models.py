# -*- coding: utf-8 -*-

import numpy as np
   
from keras.optimizers import SGD
from densenet161 import DenseNet 
from load_data_imagenet import load_data_imagenet_split
from resnet152 import resnet152_model


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



    model = resnet152_model()
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Start evaluation!")
    evaluate_model(model, weights_file_resnet, x_test, y_test, bins = 15, verbose = True)
    

    print("Evaluate DenseNet161")
    
    # Subtract mean pixel and multiple by scaling constant 
    # Reference: https://github.com/shicai/DenseNet-Caffe
    #im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
    #im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
    #im[:,:,2] = (im[:,:,2] - 123.68) * 0.017
    
    model = DenseNet(reduction=0.5, classes=1000)

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    for i in range(3):
        x_test[:,:,:,i] *= 0.017
    
    print("Evaluation for second model.")
    evaluate_model(model, weights_file_densenet, x_test, y_test, bins = 15, verbose = True)

