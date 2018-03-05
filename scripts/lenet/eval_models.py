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
from calibration import evaluate_model

batch_size    = 128
epochs        = 200
iterations    = 45000 // batch_size
num_classes10   = 10
num_classes100 = 100
weight_decay  = 0.0001
seed = 333

weights_file_10 = "../../models/lenet_c10.h5"
weights_file_100 = "../../models/lenet_c100.h5"


def build_model(num_classes = 10):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:    
        return 0.002
    return 0.0004

if __name__ == '__main__':

    print("Evaluate CIFAR.10 - LeNet")
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
        
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    
    img_mean = x_train45.mean(axis=0)  # per-pixel mean
    img_std = x_train45.std(axis=0)
    x_train45 = (x_train45-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std

    y_train45 = keras.utils.to_categorical(y_train45, num_classes10)
    y_val = keras.utils.to_categorical(y_val, num_classes10)
    y_test = keras.utils.to_categorical(y_test, num_classes10)


    # build network
    model = build_model(num_classes10)
    evaluate_model(model, weights_file_10, x_test, y_test, bins = 15, verbose = True)

    
    print("Evaluate CIFAR.100 - LeNet")
    # load data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
        
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    
    img_mean = x_train45.mean(axis=0)  # per-pixel mean
    img_std = x_train45.std(axis=0)
    x_train45 = (x_train45-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std

    y_train45 = keras.utils.to_categorical(y_train45, num_classes100)
    y_val = keras.utils.to_categorical(y_val, num_classes100)
    y_test = keras.utils.to_categorical(y_test, num_classes100)


    # build network
    model = build_model(num_classes100)
    evaluate_model(model, weights_file_100, x_test, y_test, bins = 15, verbose = True)
