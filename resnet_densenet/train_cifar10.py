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


model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, weight_decay=1e-4)
print("Model created")

model.summary()
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)  # dampening = 0.9?
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train45, x_val, Y_train45, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=seed)  # random_state = seed


#For data preprocessing, we normalize the data using the channel means and standard deviations (https://arxiv.org/pdf/1608.06993v3.pdf)

img_mean = X_train45.mean(axis=0)  # per-pixel mean
img_std = X_train45.std(axis=0)  # std
X_train45 = (X_train45-img_mean)/img_std
x_val = (x_val-img_mean)/img_std
X_test = (X_test-img_mean)/img_std


img_gen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.125,  # 0.125*32 = 4 so max padding of 4 pixels, as described in paper.
    height_shift_range=0.125
)

Y_train45 = np_utils.to_categorical(Y_train45, nb_classes)  # 1-hot vector
y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

img_gen.fit(X_train45, seed=seed)

callbacks = [LearningRateScheduler(scheduler)]


hist = model.fit_generator(img_gen.flow(X_train45, Y_train45, batch_size=batch_size, shuffle=True),
                    steps_per_epoch=len(X_train45) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(x_val, y_val),
                    validation_steps=x_val.shape[0] // batch_size, verbose=1)
                    

model.save('weights_densenet_16_8.h5')

print("Get test accuracy:")
loss, accuracy = resnet.evaluate(X_test, Y_test, verbose=0)
print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

print("Pickle models history")
with open('hist_densenet_16_8.p', 'wb') as f:
    pickle.dump(hist.history, f)