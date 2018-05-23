# Load in model weights and evaluate its goodness (ECE, MCE, error) also saves logits.

import keras
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
import wide_residual_network as wrn
from keras import regularizers
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

depth              = 34  # 32, if ignoring conv layers carrying residuals, which are needed for increasing filter size.
growth_rate        = 10  # Growth factor
n                  = (depth-4)//6
num_classes        = 100
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 45000 // batch_size
weight_decay       = 0.0005
seed = 333
weights_file_100 = "../../models/resnet_wide_28_10_c100.h5"
    
# Preprocessing based on the paper http://arxiv.org/abs/1605.07146
# and their code https://github.com/szagoruyko/wide-residual-networks
# Per channel mean and std normalization
def color_preprocessing(x_train, x_val, x_test):
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')    
    x_test = x_test.astype('float32')
    
    mean = np.mean(x_train, axis=(0,1,2))  # Per channel mean
    std = np.std(x_train, axis=(0,1,2))
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    
    return x_train, x_val, x_test  

    

# Main method
if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    # color preprocessing
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    x_train45, x_val, x_test = color_preprocessing(x_train45, x_val, x_test)    
    
    y_train45 = keras.utils.to_categorical(y_train45, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))    
    model = wrn.create_wide_residual_network(img_input, nb_classes=num_classes, N=n, k=growth_rate, dropout=0.0)
    # set optimizer
    evaluate_model(model, weights_file_100, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = "probs_resnet_wide32_c100", x_val = x_val, y_val = y_val)


