# Load in data of Birds Dataset (CUB_200_2011)
# coding: utf-8


import scipy.io
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from PIL import Image

# Paths to files, change if necessary
LABELS_PATH = '../../data/data_birds/CUB_200_2011/image_class_labels.txt'
TRAIN_TEST_SPLIT_PATH = '../../data/data_birds/CUB_200_2011/train_test_split.txt'

DATA_PATH = '../../data/data_birds/CUB_200_2011/images/'


# Load in images
def load_img(path, size = (256, 256)):
    im = Image.open(path)
    im = im.resize(size, Image.ANTIALIAS)
    rgb_im = im.convert('RGB')  # Some imageses are in Grayscale
    return np.array(rgb_im)

    
# Get center of image array
def center_crop(img_mat, size = (224, 224)):
    w,h,c = img_mat.shape
    start_h = h//2-(size[1]//2)  # Size[1] - h of cropped image
    start_w = w//2-(size[0]//2)  # Size[0] - w of cropepd image
    return img_mat[start_w:start_w+size[0],start_h:start_h+size[1], :]

    
# Load in all the Birds Data set images as an numpy array.
    # Returns: ((x_train, y_train), (x_test, y_test))
def load_data_birds(size = (256, 256), size_crop = (224, 224)):


    # ### Get test and train labels

    # First get all the labels
    labels = np.loadtxt(fname=LABELS_PATH, dtype="int16")
    y_labels = labels[:,1]  # Get only second columnt of matrix


    # Secondly get train test split of images and labels
    train_test_split = np.loadtxt(fname=TRAIN_TEST_SPLIT_PATH, dtype="int16")
    train_test_split = train_test_split[:, 1]


    # Get train and test indices
    train_idxs = np.where(train_test_split == 1)[0]
    test_idxs = np.where(train_test_split == 0)[0]


    # Split labels into train and test based on indices
    y_train = y_labels[train_idxs]
    y_test = y_labels[test_idxs]


    # ### Load in images as numpy array

    # First get folders
    # DATA_PATH
    folders = listdir(DATA_PATH)  # TODO check is folder
    imgs = []

    # Get paths to all the images
    for folder in folders:  
        path = DATA_PATH + folder + "/"
        
        # TODO load in images
        imgs.extend([join(path, f) for f in listdir(path) if isfile(join(path, f))])

    # Add image names into numpy array
    imgs = np.array(imgs)


    # Split train and test images
    train_imgs = imgs[train_idxs]
    test_imgs = imgs[test_idxs]


    # Fill in x_train array with train data
    len_train = len(train_imgs)
    # Init numpy array
    x_train = np.empty((len_train, size[0], size[1], 3), dtype="float32")

    # Load train images into numpy array
    for i, img_path in enumerate(train_imgs):
        x_train[i] = load_img(img_path, size = size)


    # ###Fill in x_test array with test data

    len_test = len(test_imgs)
    # Init numpy array
    x_test = np.empty((len_test, size_crop[0], size_crop[1], 3), dtype="float32")

    # Load in test images into array
    for i, img_path in enumerate(test_imgs):    
        img_mat = load_img(img_path, size = size)  # First scale to (256,256)
        x_test[i] = center_crop(img_mat, size = size_crop)  # Crop center of the image


    return ((x_train, y_train), (x_test, y_test))

