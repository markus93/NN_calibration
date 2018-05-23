# Loading in Stanford Cars Dataset data

import scipy.io
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

# Paths to files, change if necessary
TEST_LABELS_PATH = '../../data/data_cars/cars_test_annos_labels.mat'
TRAIN_LABELS_PATH = '../../data/data_cars/cars_train_annos.mat'
TRAIN_DATA_PATH = "../../data/data_cars/cars_train/"  # Folder full of images
TEST_DATA_PATH = "../../data/data_cars/cars_test/"


def load_img(path, new_size = 256):
    """
    Loads in an image, and converts it so its sorter side will match to new side
    
    params:
        path: (string) location to the image
        new_size: (int) the size of the image's shorter side
    returns:
        img_mat (nd.array) image matrix with shape of (width, height, channels)
    """
    
    im = Image.open(path)
    if im.size[0] < im.size[1]:
        size_perc = new_size/im.size[0]
    else:
        size_perc = new_size/im.size[1]
        
    size = (int(round(im.size[0]*size_perc, 0)), int(round(im.size[1]*size_perc, 0))) # New size of the image

    im = im.resize(size, Image.ANTIALIAS)
    rgb_im = im.convert('RGB')  # Some images are in Grayscale
    return np.array(rgb_im, dtype="float32")

 
def center_crop(img_mat, size = (224, 224)):
    """
    Center Crops an image with certain size, image must be bigger than crop size (add check for that)
    
    params:
        img_mat: (3D-matrix) image matrix of shape (width, height, channels)
        size: (tuple) the size of crops (width, height)
    returns:
        img_mat: that has been center cropped to size of center crop
    """

    w,h,c = img_mat.shape
    start_h = h//2-(size[1]//2)  # Size[1] - h of cropped image
    start_w = w//2-(size[0]//2)  # Size[0] - w of cropepd image
    return img_mat[start_w:start_w+size[0],start_h:start_h+size[1], :]


def load_data_cars(size = 256, size_crop = (224, 224)):
    """
    Main function needed to load in cars (needs rather large amount of memory)
    
    Params:
        size - image converted so its shorter side is with given size
        size_crop - test images center cropped into "size_crop"
        
    Returns:
        ((x_train, y_train), (x_test, y_test)), train and test images with class labels.
    """

    # Path to data - change according your paths
    test_labels = scipy.io.loadmat(TEST_LABELS_PATH)  # Labels saved as matlab mat-s
    train_labels = scipy.io.loadmat(TRAIN_LABELS_PATH)


    # Get labels from Matlab matrix
    test_labels = np.array(test_labels.get('annotations'))
    train_labels = np.array(train_labels.get('annotations'))


    # ### Get test and train labels

    # Length of test and train sets
    len_test = len(test_labels[0])
    len_train = len(train_labels[0])

    y_test = np.empty(len_test, dtype="int16")
    y_train = np.empty(len_train, dtype="int16")

    # Test labels
    for i in range(len_test):
        y_test[i] = test_labels[0][i][4][0][0]  # Get labels out of annotations

    # Train labels
    for i in range(len_train):
        y_train[i] = train_labels[0][i][4][0][0]

        
    # Labels start from 1, but we want it to be 0, so we could use 1-hot vector
    y_test -= 1  # min label zero, max 195
    y_train -= 1

        
    ### Load in images as numpy array
    path = TRAIN_DATA_PATH
    train_imgs = [f for f in listdir(path) if isfile(join(path, f))]
    path2 = TEST_DATA_PATH
    test_imgs = [f for f in listdir(path2) if isfile(join(path2, f))]


    # Fill in x_train array with train data
    x_train = np.empty((len_train, size, size, 3), dtype="float32")

    for i, img_path in enumerate(train_imgs):
        img_mat = load_img(TRAIN_DATA_PATH + img_path, new_size = size)  # First load and rescale image
        x_train[i] = center_crop(img_mat, size = (size, size))  # Second center crop the scaled image

    # Fill in x_test array with test data
    x_test = np.empty((len_test, size_crop[0], size_crop[1], 3), dtype="float32")

    for i, img_path in enumerate(test_imgs):    
        img_mat = load_img(TEST_DATA_PATH + img_path, new_size = size)  # First scale to 256-by-x
        x_test[i] = center_crop(img_mat, size = size_crop)  # Crop center of the image


    return ((x_train, y_train), (x_test, y_test))

