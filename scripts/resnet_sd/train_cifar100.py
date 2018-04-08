import numpy as np
import collections
import pickle

from resnet_sd import resnet_sd_model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# Learning rate schedule
def lr_sch(epoch):
    if epoch < nb_epochs * 0.5:
        return 0.1
    elif epoch < nb_epochs * 0.75:
        return 0.01
    else:
        return 0.001
        
        
# Per channel mean and std normalization
def color_preprocess(x_train, x_val, x_test):
    
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
    nb_classes = 100
    seed = 333


    # data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    
    # Data splitting (get additional 5k validation set)
    # Sklearn to split
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    x_train45, x_val, x_test = color_preprocessing(x_train45, x_val, x_test)

    # Mean per channel
    mean = np.mean(x_train45, axis=0, keepdims=True)
    print("Mean shape:", mean.shape)
    std = np.std(x_train45)
    x_train45 = (x_train45 - mean) / std
    x_val = (x_val -  mean) / std
    x_test = (x_test - mean) / std


    img_gen = ImageDataGenerator(
        horizontal_flip=True,
        data_format="channels_last",
        width_shift_range=0.125,  # 0.125*32 = 4 so max padding of 4 pixels, as described in paper.
        height_shift_range=0.125,
        fill_mode="constant",
        cval = 0
    )

    img_gen.fit(x_train45)
    
    y_train45 = np_utils.to_categorical(y_train45, nb_classes)  # 1-hot vector
    y_val = np_utils.to_categorical(y_val, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
        
    # building and training net
    model = resnet_sd_model(img_shape = (32,32), img_channels = 3, 
                            layers = 110, nb_classes = nb_classes, verbose = True)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit_generator(img_gen.flow(x_train45, y_train45, batch_size=batch_size, shuffle=True),
                    steps_per_epoch=len(x_train45) // batch_size,
                    validation_steps=len(x_val) // batch_size,
                    epochs=nb_epochs,
                    validation_data = (x_val, y_val),
                    callbacks=[LearningRateScheduler(lr_sch)])

    model.save_weights('model_weight_ep500_110SD_cifar_100.hdf5')
    
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_110SD_cifar100.p', 'wb') as f:
        pickle.dump(hist.history, f)
    
    

