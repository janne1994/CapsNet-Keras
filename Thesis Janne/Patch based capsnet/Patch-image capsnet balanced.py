# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras import layers
from keras import models
from keras.models import Sequential

import h5py
from pathlib import Path
import os

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import layers, models, optimizers
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction import image
from skimage import color

# Make sure the capsulelayers package is installed from: https://github.com/XifengGuo/CapsNet-Keras


"""## Loading images into Python and displaying them"""
wd = os.getcwd()
os.chdir('/content/drive/My Drive/Colab Notebooks/Thesis/')


#https://realpython.com/storing-images-in-python/#storing-with-hdf5
def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(f"{num_images}.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

yes_img, yes_lab = read_many_hdf5('pos_patch_labs155') #positive patches only for labeling
no_img, no_lab = read_many_hdf5('pos_patch_labs98')

import matplotlib.pyplot as plt
brain_tumor_tst = yes_img[50:70]
i = 0
images_num = np.arange(155)
plt.figure(figsize = (25,25))
for image,lab in zip(brain_tumor_tst,images_num):
    plt.subplot(4,5,i+1)
    i += 1
    plt.axis('off') # turning off the axes
    plt.title(lab)
    plt.imshow(image, cmap = 'gray')

"""### Converting to grayscale, and removing duplicates"""

from skimage import color
yes_crop = color.rgb2gray(yes_img)
no_crop = color.rgb2gray(no_img)

"""### Dividing up into patches"""

def blockshaped(arr, nrows, ncols): 
    """
	FROM https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    patched_imgs = []
    for image in arr:
        h, w = image.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        patched = image.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)
        patched_imgs.append(patched)
    return np.array(patched_imgs)

"""### Reading in the manual labeling, loading in the right format and lining up the numbers with the patches"""

def conv_int_list(rows_list):
    """ removes NA values, a list of every cell, and every row and then the final list
    converts to integer values"""
    list_all = []
    for i in rows_list:
        list_row = []
        for j in i:
            if j != '0':
                list_row.append(j.split(','))
        list_all.append(list_row)
        integer_list= [[[int(j)for j in i]for i in k]for k in list_all]
    return integer_list

def labeling_patches(patch_ranges):
    all_labels = []
    for idx in patch_ranges:
        labels = np.zeros((256))
        for subidx in idx:
            if len(subidx) == 2:
                labels[subidx[0]:subidx[1]+1] = 1
            else:
                labels[subidx[0]+1] = 1
        all_labels.append(labels)
    return np.array(all_labels)

def labeling_neg_patches(patches_arra):
  labeled = []
  for patch in patches_arra:
    labels = np.zeros((256))
    labeled.append(labels)
  return np.array(labeled)

x_patches_yes = blockshaped(yes_crop, 14, 14)
x_patches_no = blockshaped(no_crop, 14, 14)

print(x_patches_yes.shape)
print(x_patches_no.shape)

plt.imshow(x_patches_yes[0][40])

#Patches indicating where a tumor is - manually labeled and imported from excel
import pandas as pd
df = pd.read_excel('/content/drive/My Drive/Colab Notebooks/Thesis/manual label thesis - duplrem.xlsx', header = 0).iloc[0:, 1:]
df = df.fillna('0')
t = df.values.tolist() #convert to list
label_ranges = conv_int_list(t) #Building the right list type for the labels

x_labels_yes = labeling_patches(label_ranges) #labeling the patches in the right place
x_labels_no = labeling_neg_patches(x_patches_no)
flattened_yes_labs = x_labels_yes.reshape((39680))
flattened_yes_patches = x_patches_yes.reshape((39680, 14, 14))
flattened_no_labs = x_labels_no.reshape((25088))
flattened_no_patches = x_patches_no.reshape((25088, 14, 14))

print(flattened_yes_labs.shape)
print(flattened_yes_patches.shape)

print(flattened_no_labs.shape)
print(flattened_no_patches.shape)

#Displaying a patched image
i = 0
plt.figure(figsize = (18,14))
for image, num in zip(flattened_yes_patches[256:512], (flattened_yes_labs[256:512].astype('uint8'))):
    plt.subplot(16,16,i+1)
    i += 1
    plt.axis('off') # turning off the axes
    plt.title(num)
    plt.imshow(image, cmap = 'gray')

"""### Combining the labels and splitting the sets"""

flattened_yes_labs = list(flattened_yes_labs)
flattened_no_labs = list(flattened)

all_patches = np.vstack((flattened_yes_patches, flattened_no_patches))
all_labs = np.array(list(list(flattened_yes_labs) + list(flattened_no_labs)))

print(all_patches.shape)
print(all_labs.shape)

from sklearn.model_selection import train_test_split
x_train1,  x_test, y_train1, y_test = train_test_split(all_patches, all_labs, test_size = 0.20, random_state = 1)
x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size = 0.2, random_state = 2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)


"""Next, the code for balancing the datasets"""
def balanced_subsample(x,y,subsample_size=1.0):
	"""adapted from https://stackoverflow.com/questions/23455728/scikit-learn-balanced-subsampling"""

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

balanced_x_train, balanced_y_train = balanced_subsample(x_train, y_train, 1666)
balanced_x_test, balanced_y_test = balanced_subsample(x_test, y_test, 475)
balanced_x_val, balanced_y_val = balanced_subsample(x_val, y_val, 359)

print(balanced_x_train.shape)
print(balanced_x_test.shape)
print(balanced_x_val.shape)

#Shaping the patches so they fit into the model
x_train_patches = x_train.reshape((3332, 14, 14 ,1))
x_test_patches = x_test.reshape((950, 14, 14 ,1))
x_val_patches = x_val.reshape((718, 14, 14 ,1))
y_train_patches = y_train
y_val_patches = y_val
y_test_patches = y_test
print(x_train_patches.shape)
print(x_val_patches.shape)
print(x_test_patches.shape)

"""#### Capsnet, balanced"""

def CapsNet(input_shape, n_class, routings):

   x = layers.Input(shape=(input_shape))

   # Layer 1: Just a conventional Conv2D layer
   conv1 = Conv2D(filters=256, kernel_size=9, strides=1, padding='same', activation='relu', name='conv1')(x)
   
   # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
   primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='same')

   # Layer 3: Capsule layer. Routing algorithm works here.
   digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
   name='digitcaps')(primarycaps)

   # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
   # If using tensorflow, this will not be necessary. :)
   out_caps = Length(name='capsnet')(digitcaps) # CAN WE EXCLUDE THIS IN KERAS TOO?

   # Decoder network.
   y = layers.Input(shape=(n_class,))
   masked_by_y = Mask()([digitcaps, y]) # The true label is used to mask the output of capsule layer. For training
   masked = Mask()(digitcaps) # Mask using the capsule with maximal length. For prediction

   # Shared Decoder model in training and prediction
   decoder = models.Sequential(name='decoder')
   decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class)) # YES
   decoder.add(layers.Dense(1024, activation='relu')) # YES
   decoder.add(layers.Dense(4096, activation = 'relu'))
   decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
   decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

   # Models for training and evaluation (prediction)
   train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
   eval_model = models.Model(x, [out_caps, decoder(masked)])

   return train_model, eval_model
  
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

model, eval_model = CapsNet(input_shape=((14,14,1)), n_class=1, routings=2)

# compile the model
model.compile(optimizer=optimizers.Adam(lr=3e-3),
 loss=[margin_loss, 'mse'],
 metrics={'capsnet': 'accuracy'})

model.summary()

# Fitting the model
history = model.fit(
        [x_train_patches, y_train_patches],[y_train_patches,x_train_patches],
        batch_size=128,
        epochs=30,
        validation_data=([x_val_patches, y_val_patches], [y_val_patches, x_val_patches])) 

#Acc & loss plots
plt.plot(history.history['capsnet_acc'])
plt.plot(history.history['val_capsnet_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['capsnet_loss'])
plt.plot(history.history['val_capsnet_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save_weights('capsnet_patches_final_7jan.h5')

# Metrics
y_pred, x_recon = eval_model.predict(x_test_patches)

predictions = [1 if  x>0.5 else 0 for x  in y_pred]
print(np.array(predictions).shape)

confusion_mtx = confusion_matrix(y_test_patches, predictions) 
print(confusion_mtx)

print('f1',f1_score(y_test_patches, predictions ))
print(classification_report(y_test_patches, predictions))
print('acc', accuracy_score(y_test_patches, predictions))
