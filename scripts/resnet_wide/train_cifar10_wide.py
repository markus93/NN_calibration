import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Model
from keras import optimizers
from keras import regularizers
from sklearn.model_selection import train_test_split
import pickle

depth              = 16
wide               = 8
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 45000 // batch_size
weight_decay       = 0.0005
log_filepath       = r'./w_resnet/'
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
def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test

def wide_residual_network(img_input,classes_num,depth,k):

    print('Wide-Resnet %dx%d' %(depth, k))
    n_filters  = [16, 16*k, 32*k, 64*k]
    n_stack    = (depth - 4) / 6
    in_filters = 16

    def conv3x3(x,filters):
    	return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
    	kernel_initializer=he_normal(),
        kernel_regularizer=regularizers.l2(weight_decay))(x)

    def residual_block(x,out_filters,increase_filter=False):
        if increase_filter:
            first_stride = (2,2)
        else:
            first_stride = (1,1)
        pre_bn   = BatchNormalization()(x)
        pre_relu = Activation('relu')(pre_bn)
        conv_1 = Conv2D(out_filters,kernel_size=(3,3),strides=first_stride,padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1   = BatchNormalization()(conv_1)
        relu1  = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        if increase_filter or in_filters != out_filters:
            projection = Conv2D(out_filters,kernel_size=(1,1),strides=first_stride,padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)
            block = add([conv_2, projection])
        else:
            block = add([conv_2,x])
        return block

    def wide_residual_layer(x,out_filters,increase_filter=False):
    	x = residual_block(x,out_filters,increase_filter)
    	in_filters = out_filters
    	for _ in range(1,int(n_stack)):
    		x = residual_block(x,out_filters)
    	return x


    x = conv3x3(img_input,n_filters[0])
    x = wide_residual_layer(x,n_filters[1])
    x = wide_residual_layer(x,n_filters[2],increase_filter=True)
    x = wide_residual_layer(x,n_filters[3],increase_filter=True)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed

    y_train45 = keras.utils.to_categorical(y_train45, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output = wide_residual_network(img_input,num_classes,depth,wide)
    model = Model(img_input, output)
    print(model.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]

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
    model.save('resnet_wide_16_8_c10.h5')
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_wide_16_8_cifar10.p', 'wb') as f:
        pickle.dump(hist.history, f)