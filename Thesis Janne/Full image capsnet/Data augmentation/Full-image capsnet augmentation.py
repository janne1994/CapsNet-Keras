# -*- coding: utf-8 -*-
!git clone https://github.com/XifengGuo/CapsNet-Keras.git

!pip install --user keras==2.2.4
!pip install --user tensorflow
!pip install --user cv2


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

plt.imshow(x_val[0])

from skimage import color
x_train = color.rgb2gray(x_train1)
x_test = color.rgb2gray(x_test1)
x_val = color.rgb2gray(x_val1)

"""Resize to model can train"""
def resizing_more(data):
  new_list = []
  for i in data:
    new = cv2.resize(i, (64,64))
    new_list.append(new)
  return np.array(new_list)

x_train = resizing_more(x_train)
x_test = resizing_more(x_test)
x_val = resizing_more(x_val)


"""Adapted from Xifeng Guo (GITHUB)"""
path = wd + '/drive/My Drive/Colab Notebooks/Thesis/'
os.chdir(path)
np.random.seed(4)


K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
   x = layers.Input(shape=input_shape)

   # Layer 1: Just a conventional Conv2D layer
   conv1 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
   conv2 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
   
   # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
   primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

   # Layer 3: Capsule layer. Routing algorithm works here.
   digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
   name='digitcaps')(primarycaps)

   # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
   # If using tensorflow, this will not be necessary. :)
   out_caps = Length(name='capsnet')(digitcaps)

   # Decoder network.
   y = layers.Input(shape=(n_class,))
   masked_by_y = Mask()([digitcaps, y]) # The true label is used to mask the output of capsule layer. For training
   masked = Mask()(digitcaps) # Mask using the capsule with maximal length. For prediction

   # Shared Decoder model in training and prediction
   decoder = models.Sequential(name='decoder')
   decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class)) # YES
   decoder.add(layers.Dense(1024, activation='relu')) # YES
   decoder.add(layers.Dense(4096, activation='relu')) # YES
   decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
   decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

   # Models for training and evaluation (prediction)
   train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
   eval_model = models.Model(x, [out_caps, decoder(masked)])

   # manipulate model
   noise = layers.Input(shape=(n_class, 16))
   noised_digitcaps = layers.Add()([digitcaps, noise])
   masked_noised_y = Mask()([noised_digitcaps, y])
   manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

   return train_model, eval_model, manipulate_model
  
  
  
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


#Reshaping so I can feed it to the network
x_train = x_train.reshape((2995,64,64,1))
x_val = x_val.reshape((749,64,64,1))
x_test = x_test.reshape((40, 64, 64, 1))

model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
 n_class=1,
 routings=2)

# compile the model
model.compile(optimizer=optimizers.Adam(lr=3e-3),
 loss=[margin_loss, 'mse'],
 metrics={'capsnet': 'accuracy'})

model.summary()

#Fitting the model
history = model.fit(
        [x_train, y_train],[y_train,x_train],
        batch_size=128,
        epochs=30,
        validation_data=([x_val, y_val], [y_val, x_val]), 
        shuffle=True)

print(history.history.keys())

#version 2.5
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
y_pred, x_recon = eval_model.predict(x_test)


predictions = [1 if  x>0.5 else 0 for x  in y_pred]
print(np.array(predictions).shape)


confusion_mtx = confusion_matrix(y_test, predictions) 
print(confusion_mtx)

print(f1_score(y_test, predictions ))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

