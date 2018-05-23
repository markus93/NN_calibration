# Fine-tuning procedure for Birds using ResNet 50

import keras
import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from load_data_birds import load_data_birds
from image_gen_extended import ImageDataGenerator, random_crop
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model

SIZE_IMG = 256
SIZE_CROP = (224, 224)
BATCH_SIZE = 64
NR_CLASSES = 200  # Classes for birds
EPOCHS = 250
SEED = 333  # Random seed for reproducibility
LR = 0.0001

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
    
    
    #Data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True)
    datagen.config['random_crop_size'] = SIZE_CROP
    datagen.set_pipeline([random_crop])  # Add random crop for training
    datagen.fit(x_train) 
    
    # Model loading and preparing for fine-tuning
    print("Load model")
    base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', 
                                                      input_tensor=None, input_shape=(224,224,3), 
                                                      pooling=None, classes=NR_CLASSES)  # Load in pretrained model (ImageNet)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global AVG pool and extra Dense layer can be dropped. However given solution matches the weight file.
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NR_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    
    sgd = SGD(lr=0.0001, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_birds_best.h5', verbose=1, save_best_only=True)
    cbks = [early_stopping, checkpointer]
    
    # Model training
    print("Start training")
    hist = model.fit_generator(datagen.flow(x_train, y_train, shuffle=True, batch_size=BATCH_SIZE),
                         steps_per_epoch=len(x_train) // BATCH_SIZE,
                         epochs=EPOCHS,
                         callbacks=cbks,
                         validation_data=(x_val, y_val))
                         
    model.save('resnet50_birds.h5')
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test50, y_test50, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_resnet50_birds.p', 'wb') as f:
        pickle.dump(hist.history, f)