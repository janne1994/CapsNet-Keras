import numpy as np
from skimage import io, color
import h5py
import matplotlib.pyplot as plt
import glob
import imutils
from sklearn.model_selection import train_test_split

import os
import cv2


direc = (os.getcwd()+ "\\xyz\\")

def get_files(folder):
    images = []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        if img is not None:
            images.append(img)
    return np.array(images)

img_pos = get_files(direc + '\\yes')
img_neg = get_files(direc + '\\no')

def add_1_labels(folder):
    """ importing from specific folders only"""
    labels = []
    for i in get_files(folder):
        labels.append(int(1))
    return np.array(labels)

def add_0_labels(folder):
    labels = []
    for i in get_files(folder):
        labels.append(int(0))
    return np.array(labels)
        
y_labels_yes = add_1_labels(direc+'\\yes')
y_labels_no = add_0_labels(direc+'\\no')

print("we have {} cases with tumor present".format(len(y_labels_yes))) 
print("we have {} cases with no tumor present".format(len(y_labels_no)))

#Downsampling
x_pos_downsample = img_pos[:98]
x_neg_downsample = img_neg
y_labels_yes = y_labels_yes[:98]
print("After downsampling we have {} cases with tumor present".format(len(x_pos_downsample)))
print("After downsampling we have {} cases with no tumor present".format(len(x_neg_downsample)))

data_tumor = np.concatenate((x_neg_downsample, x_pos_downsample))
data_label = np.concatenate((y_labels_no, y_labels_yes))


# ## Edge detection to crop
# We crop to eliminate the black space, and to be able to nicely resize the images later.
#With thanks to https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/

def crop_imgs(data):
    cropped_images = []
    for image in data:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # determine the most extreme points along the contour
        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])

        cropped = image[top[1] : bottom[1], left[0] : right[0]].copy()
        edited = cv2.resize(cropped, (224,224))
        cropped_images.append(edited)
    return np.array(cropped_images)



# ## Cropping the images
cropped_imgs = crop_imgs(data_tumor)

#Split on train and test.
x_train, x_test, y_train, y_test = train_test_split(cropped_imgs, data_label, test_size=0.2, 
                                                    random_state=2, shuffle = True, stratify = data_label)


# ## Data augmentation by rotation
augmented_set = x_train
new_set = []

for angle in np.arange(0, 360, 15): #24 positions
    for image in augmented_set: #applied to each image
        rotated = imutils.rotate(image, angle)
        new_set.append(rotated)
        
new_y_set = []

#Same for labels
for angle in np.arange(0, 360, 15):
    new_y_set = np.append( new_y_set, y_train, axis = 0)

x_train = np.array(new_set)
y_train = np.array(new_y_set)

print(x_train.shape, y_train.shape)

#Split the train set into the train set and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, 
                                                  random_state=2, shuffle = True, stratify = y_train)


#Storing the images as HDF5 file
from pathlib import Path
wd = os.getcwd()
hdf5_dir = Path(wd + "/xyz/")

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()


train = store_many_hdf5(x_train, y_train)
val = store_many_hdf5(x_val, y_val)
test = store_many_hdf5(x_test, y_test)