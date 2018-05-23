# Training procedure for CIFAR-10 using DenseNet 40 with growth rate 12.

from __future__ import print_function

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from sklearn.model_selection import train_test_split


batch_size = 64
nb_classes = 10
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


def scheduler(epoch):
    if epoch < nb_epoch/2:
        return learning_rate
    elif epoch < nb_epoch*3/4:
        return learning_rate*0.1
    return learning_rate*0.01
    
    
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



model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, weight_decay=1e-4)
print("Model created")

model.summary()
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#For data preprocessing, we normalize the data using the channel means and standard deviations (https://arxiv.org/pdf/1608.06993v3.pdf)
x_train, x_test = color_preprocessing(x_train, x_test)

x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed


img_gen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.125,  # 0.125*32 = 4 so max padding of 4 pixels, as described in paper.
    height_shift_range=0.125,  # first zero-padded with 4 pixels on each side, then randomly cropped to again produce 32×32 images
    fill_mode = "constant",
    cval = 0
)

y_train45 = np_utils.to_categorical(y_train45, nb_classes)  # 1-hot vector
y_val = np_utils.to_categorical(y_val, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

img_gen.fit(x_train45, seed=seed)

callbacks = [LearningRateScheduler(scheduler)]


hist = model.fit_generator(img_gen.flow(x_train45, y_train45, batch_size=batch_size, shuffle=True),
                    steps_per_epoch=len(x_train45) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(x_val, y_val),
                    validation_steps=x_val.shape[0] // batch_size, verbose=1)
                    

model.save('weights_densenet_16_8.h5')

print("Get test accuracy:")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

print("Pickle models history")
with open('hist_densenet_16_8.p', 'wb') as f:
    pickle.dump(hist.history, f)