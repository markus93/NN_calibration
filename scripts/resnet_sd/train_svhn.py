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


# Callbacks for updating gates and learning rate
def scheduler(epoch):

    if epoch < 30:
        return learning_rate
    elif epoch < 35:
        return learning_rate*0.1
    return learning_rate*0.01
    

if __name__ == '__main__':

    # constants
    learning_rate = 0.1
    nb_epochs = 50
    batch_size = 128
    nb_classes = 10
    seed = 333
    layers = 152 # n = 25 (152-2)/6


    # data
    print("Loading data, may take some time and memory!")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data_svhn(seed = seed)
    print(x_train.shape)
    print("Data loaded")
    
    # Simple preprocessing - subtract mean and divide by standard division
    img_mean = x_train.mean(axis=0)  # per-pixel mean
    img_std = x_train.std(axis=0)
    x_train = (x_train-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std
    
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
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy",metrics=["accuracy"])  

    print("Model compiled")

    hist = model.fit_generator(img_gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                    steps_per_epoch=len(x_train) // batch_size,
                    validation_steps=len(x_val) // batch_size,
                    epochs=nb_epochs,
                    validation_data = (x_val, y_val),
                    callbacks=[LearningRateScheduler(scheduler)])

    model.save_weights('model_weight_ep50_152SD_svhn.hdf5')
    
    
    # For evaluation should load model with pL = 1

    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_152SD_svhn.p', 'wb') as f:
        pickle.dump(hist.history, f)
    