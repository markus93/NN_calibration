# Code example from https://github.com/sebastianbk/finetuned-resnet50-keras/blob/master/resnet50_train.py

import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
import pickle
from sklearn.model_selection import train_test_split
from load_data_birds import load_data_birds
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

SIZE_IMG = 256
SIZE_CROP = (224, 224)
BATCH_SIZE = 64
NR_CLASSES = 200  # Classes for birds
EPOCHS = 250
SEED = 333  # Random seed for reproducibility
LR = 0.0001
weights_file = "../../models/resnet_50_birds.h5"

MEAN = [103.939, 116.779, 123.68]


# Train 5994 and val/test 2897

# Issue about random crop
# https://github.com/keras-team/keras/issues/3338

if __name__ == "__main__":

    print("Load data")
    
    (x_train, y_train), (x_test, y_test) = load_data_birds(SIZE_IMG, SIZE_CROP)
    
    y_train = keras.utils.to_categorical(y_train, NR_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NR_CLASSES)
    
    #  If you are freezing initial layers, you should use imagenet mean/std. (https://discuss.pytorch.org/t/confused-about-the-image-preprocessing-in-classification/3965)            
    x_train = x_train[..., ::-1]
    x_test = x_test[..., ::-1]
    
    for i in range(3):
        x_train[:,:,:,i] -= MEAN[i]
        x_test[:,:,:,i] -= MEAN[i]

    
    x_test50, x_val, y_test50, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=SEED)
    
    print("Load model")
    base_model = keras.applications.resnet50.ResNet50(include_top=False)  # Load in pretrained model (ImageNet)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # let's add a fully-connected layer
    predictions = Dense(NR_CLASSES, activation='softmax')(x)  # and a logistic layer

    model = Model(inputs=base_model.input, outputs=predictions)
    
    evaluate_model(model, weights_file, x_test50, y_test50, bins = 15, verbose = True, 
               pickle_file = "probs_resnet50_birds", x_val = x_val, y_val = y_val)
    
    