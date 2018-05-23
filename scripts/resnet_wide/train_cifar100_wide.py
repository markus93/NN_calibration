# Training procedure for WideNet 32-10 for CIFAR-100.
import keras
import numpy as np
from keras.datasets import cifar100
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

depth              = 34
growth_rate        = 10  # Growth factor
n                  = (depth-4)//6  # N, number of blocks in one group
num_classes        = 100
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 45000 // batch_size
weight_decay       = 0.0005
seed = 333

def scheduler(epoch):
    if epoch <= 60:
        return 0.1
    if epoch <= 120:
        return 0.02
    if epoch <= 160:
        return 0.004
    return 0.0008

    
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
    print(model.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    change_lr = LearningRateScheduler(scheduler)
    checkpointer = ModelCheckpoint('model_wide_28_10_c100_best.hdf5', verbose=1, save_best_only=True)
    cbks = [change_lr, checkpointer]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect') # Missing pixels replaced with reflections

    datagen.fit(x_train45)

    # start training
    hist = model.fit_generator(datagen.flow(x_train45, y_train45, batch_size=batch_size, shuffle=True),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_val, y_val))
    model.save('resnet_wide_28_10_c100.h5')
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_wide_28_10_cifar100.p', 'wb') as f:
        pickle.dump(hist.history, f)