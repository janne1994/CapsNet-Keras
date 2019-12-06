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


import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras
import keras.backend as K

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import layers, models, optimizers
from keras.applications import vgg16
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction import image
from skimage import color


"""## Loading images into Python and displaying them"""
wd = os.getcwd()
hdf5_dir = Path(wd + '/drive/My Drive/Colab Notebooks/Thesis/')


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
    file = h5py.File(hdf5_dir / f"{num_images}.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

x_train1, y_train = read_many_hdf5(161)
x_test1, y_test = read_many_hdf5(41)
x_val1, y_val = read_many_hdf5(51)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


x_train = color.rgb2gray(x_train1)
x_test = color.rgb2gray(x_test1)
x_val = color.rgb2gray(x_val1)


"""## Patches"""
def convert_to_patches_w_labels(data, labels, patch_size = (20, 20), max_patches= 250):
  new_patches = []
  for img in data:
    new_patch = image.extract_patches_2d(img, patch_size, max_patches)
    new_patches.append(new_patch)
  flattened = np.array(new_patches).reshape((data.shape[0]* max_patches, patch_size[0], patch_size[1]))
  
  new_labels = []
  for label in labels:
    new_labels.append([label]*max_patches)
  new_labels = np.array(new_labels).reshape((data.shape[0] * max_patches))
  return flattened, new_labels

x_train_patches, y_train_patches = convert_to_patches_w_labels(x_train, y_train)
x_test_patches, y_test_patches = convert_to_patches_w_labels(x_test, y_test)
x_val_patches, y_val_patches = convert_to_patches_w_labels(x_val, y_val)

print(x_train_patches.shape)
print(y_train_patches.shape)

print(x_test_patches.shape)
print(y_test_patches.shape)

print(x_val_patches.shape)
print(y_val_patches.shape)

"""### Visualization of the patches"""
idx = np.random.choice(np.arange(len(x_train_patches)), 20, replace=False)
data_tumor_viz = x_train_patches[idx]
data_tumor_lab = y_train_patches[idx]


i = 0
plt.figure(figsize = (15,8))
for image, label in zip(data_tumor_viz, data_tumor_lab):
    plt.subplot(4,5,i+1)
    i += 1
    plt.axis('off') # turning off the axes
    plt.title(label)
    plt.imshow(image, cmap = 'gray')
   

print("We have", sum(y_train_patches==0),"patches with no tumor in x_train and", sum(y_train_patches == 1), "patches with a tumor")


"""Patch based model"""

def CapsNet(input_shape, n_class, routings):
   x = layers.Input(shape=input_shape)

   # Layer 1: Just a conventional Conv2D layer
   conv1 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

   # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
   primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

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

   return train_model, eval_model#, manipulate_model
  
  
  
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


model, eval_model = CapsNet(input_shape=x_train_patches.shape[1:], 
 n_class=1,
 routings=2)

# compile the model
model.compile(optimizer=optimizers.Adam(lr=3e-3),
 loss=[margin_loss, 'mse'],
 metrics={'capsnet': 'accuracy'})

model.summary()


history = model.fit(
        [x_train_patches, y_train_patches],[y_train_patches,x_train_patches],
        batch_size=128,
        epochs=30,
        validation_data=([x_val_patches, y_val_patches], [y_val_patches, x_val_patches])) 

model.save_weights('capsnet_patches.h5')

# Do this if I want to load the model again plus weights
model, eval_model = CapsNet(input_shape=x_train_patches.shape[1:],
 n_class=1,
 routings=2)
model.summary()
model.load_weights('capsnet_patches_v2.5.h5')


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

#Metrics
y_pred, x_recon = eval_model.predict(x_test_patches, batch_size=100)

predictions = [1 if  x>0.5 else 0 for x  in y_pred]
print(np.array(predictions).shape)


confusion_mtx = confusion_matrix(y_test_patches, predictions) 
print(confusion_mtx)

print(f1_score(y_test_patches, predictions ))
print(classification_report(y_test_patches, predictions))
print(accuracy_score(y_test_patches, predictions))








