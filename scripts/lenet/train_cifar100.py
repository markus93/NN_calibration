import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pickle

batch_size    = 128
epochs        = 200
iterations    = 45000 // batch_size
num_classes   = 100
weight_decay  = 0.0001
seed = 333


def build_model(n=1, num_classes = 10):
    """
    parameters:
        n: (int) scaling for model (n times filters in Conv2D and nodes in Dense)
    """
    model = Sequential()
    model.add(Conv2D(n*12, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=(32,32,3)))
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

def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:    
        return 0.002
    return 0.0004
    
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

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = color_preprocessing(x_train, x_test)
    
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed

    
    y_train45 = keras.utils.to_categorical(y_train45, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    # build network
    model = build_model(n=2, num_classes = num_classes)
    print(model.summary())

    # set callback
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train45)

    # start traing 
    model.fit_generator(datagen.flow(x_train45, y_train45,batch_size=batch_size, shuffle=True),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_val, y_val))
    # save model
    model.save('lenet_c100.h5')
    
    print("Get test accuracy:")
    loss, accuracy = resnet.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

    print("Pickle models history")
    with open('hist_lenet_c100.p', 'wb') as f:
        pickle.dump(hist.history, f)