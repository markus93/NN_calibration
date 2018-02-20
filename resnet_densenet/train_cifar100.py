from __future__ import print_function

import sys
sys.setrecursionlimit(10000)

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split


batch_size = 64
nb_classes = 100
nb_epoch = 300  # EDIT: was 15; Is 15 enough? Try both?

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 12
bottleneck = False
reduction = 0.0
dropout_rate = 0.0 # 0.0 for data augmentation
seed = 333


model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                          bottleneck=bottleneck, reduction=reduction, weights=None)
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train45, x_val, Y_train45, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=seed)  # random_state = seed


img_mean = X_train45.mean(axis=0)  # per-pixel mean
X_train45 = X_train45-img_mean
x_val = x_val-img_mean
X_test = X_test-img_mean


img_gen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.125,  # 0.125*32 = 4 so max padding of 4 pixels, as described in paper.
    height_shift_range=0.125
)

Y_train45 = np_utils.to_categorical(Y_train45, nb_classes)  # 1-hot vector
y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

img_gen.fit(X_train, seed=seed)

# Load model
# model.load_weights("weights/DenseNet-BC-100-12-CIFAR100.h5")
# print("Model loaded.")

lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
model_checkpoint= ModelCheckpoint("weights/DenseNet-40-12-CIFAR100.h5", monitor="val_acc", save_best_only=True,
                                  save_weights_only=True)

callbacks=[lr_reducer, early_stopper, model_checkpoint]


model.fit_generator(img_gen.flow(X_train45, Y_train45, batch_size=batch_size, shuffle=True), samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                   callbacks=callbacks,
                   validation_data=(x_val, y_val),
                   nb_val_samples=x_val.shape[0], verbose=1)

yPreds = model.predict(X_test)
yPred = np.argmax(yPreds, axis=1)
yTrue = Y_test

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)