# -*- coding: utf-8 -*-
"""V1.0 K-fold CNN.ipynb
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import h5py
from pathlib import Path
import os

import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import AveragePooling2D, MaxPooling2D

from sklearn.model_selection import StratifiedKFold
import datetime
from keras.models import save_model
from tensorflow.python.keras.models import Model, load_model, save_model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score


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

#Augmented data
x_train, y_train = read_many_hdf5(2995)
x_test, y_test = read_many_hdf5(40)
x_val, y_val = read_many_hdf5(749)

#Concatenate for k-fold cross validation
x_trainz = np.concatenate((x_train, x_val))
y_trainz = np.concatenate((y_train, y_val))

x_train = x_trainz
y_train = y_trainz


"""## Normalizing the images."""
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


"""## What images do we have?
"""
data_tumor_viz = x_train[:10]
i = 0
plt.figure(figsize = (25,25))
for image in data_tumor_viz:
    plt.subplot(4,5,i+1)
    i += 1
    plt.axis('off') # turning off the axes
    plt.imshow(image, cmap = 'gray')

for image in data_tumor_viz:
    print(image.shape)

print("We have", sum(y_train==0),"with no tumor in x_train and", sum(y_train == 1), "with a tumor")

#Modeling
def CNN_model():
  adam = Adam(lr=0.0001)  #LR of 0.001 LEADS TO OVERFITTING
  model = Sequential()
  model.add(layers.Conv2D(32,(4,4), activation = 'relu', input_shape = (224,224,3), name = 'conv1')) 
  model.add(layers.MaxPooling2D((2, 2), name = 'maxpool1'))
  model.add(layers.Conv2D(64, (4,4), activation = 'relu', name = 'conv2'))
  model.add(layers.MaxPooling2D((2, 2), name = 'maxpool2'))
  model.add(layers.Conv2D(64, (4,4), activation = 'relu', name = 'conv3')) #added1
  model.add(layers.MaxPooling2D((2, 2), name = 'maxpool3')) #added1
  model.add(layers.Flatten(name = 'flatten'))
  model.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.001), name = 'dense1'))
  #REMOVED BATCHNORM
  model.add(layers.Dropout(0.5, name = 'dropout'))
  model.add(layers.Dense(1, activation = 'sigmoid', name = 'dense2')) #SOFTMAX DOESNT TRAIN SOMEHOW, SO STICK WITH SIGMOID
	# Compile model
  model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model


model = CNN_model()
model.summary()

k = 5
folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_train, y_train))


import os
os.chdir('/content/drive/My Drive/Colab Notebooks/Submission 6dec thesis/')

cvscores = []
for j, (train_idx, val_idx) in enumerate(folds,1):
    
    print('\nFold ',j)
    X_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = x_train[val_idx]
    y_valid_cv= y_train[val_idx]
    
    model = CNN_model()
    history = model.fit(x_train[train_idx], y_train[train_idx], epochs=30, batch_size=32, validation_data = (X_valid_cv, y_valid_cv), shuffle = True)
    model.save('CNN_model_'+ str(j) + '.h5')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    scores = model.evaluate(X_valid_cv, y_valid_cv)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Load the model from the best fold
wd = os.getcwd()

modelfold1 = load_model('CNN_model_1.h5')
modelfold2 = load_model('CNN_model_2.h5')
modelfold3 = load_model('CNN_model_3.h5')
modelfold4 = load_model('CNN_model_4.h5')
modelfold5 = load_model('CNN_model_5.h5')



def multiple_predictions(test_data, list_of_models):
  predicts = []
  for i in list_of_models:
    predict_fold = i.predict(test_data)
    labeled = [1 if x > 0.5 else 0 for x in predict_fold]
    predicts.append(labeled)
  return predicts

def multiple_accuracies(test_labs, list_of_model_predictions):
  accuracies = []
  fold = 0
  for i in list_of_model_predictions:
    fold +=1
    accuracy = accuracy_score(test_labs, i)
    accuracies.append(accuracy)
  return accuracies

def multiple_conf_matrices(test_labs, list_of_model_predictions):
  confusions = []
  for i in list_of_model_predictions:
    confusion_mt = confusion_matrix(test_labs, i)
    confusions.append(confusion_mt)
  return confusions

def multiple_class_reports(test_labs, list_of_model_predictions):
  class_reps = []
  for i in list_of_model_predictions:
    class_rep = classification_report(test_labs, i)
    class_reps.append(class_rep)
  return class_reps

preds = multiple_predictions(x_test, [modelfold1, modelfold2, modelfold3, modelfold4, modelfold5])
accu = multiple_accuracies(y_test, preds)
conf = multiple_conf_matrices(y_test, preds)
clas = multiple_class_reports(y_test, preds)

fold = 0
for i in accu:
  fold +=1
  print ('for fold', fold, 'the accuracy is', i)
print('the mean accuracy is ', round(np.mean(accu),2)*100, '%')

fold = 0
for i in conf:
  fold +=1
  print('\nfor fold', fold, 'the confusion matrix is\n', i)

fold = 0
for i in clas:
  fold +=1
  print('\nfor fold', fold, 'the classification report is\n', i )

from sklearn.metrics import precision_score, recall_score
precscores = []
recscores = []
f1scores = []
for i in preds:
  prec = precision_score(y_test, i)
  rec = recall_score(y_test,i)
  f1 = f1_score(y_test,i)
  precscores.append(prec)
  recscores.append(rec)
  f1scores.append(f1)
  
print("the average precision score is", round(np.mean(precscores),2)*100, '%')
print("the average recall is", round(np.mean(recscores),2)*100, '%')
print("the average F1 score is", round(np.mean(f1scores),2)*100, '%')

#Image classifications
i = 0
plt.figure(figsize = (25,25))
for image in x_test:
    plt.subplot(10,5,i+1)
    plt.axis('off') # turning off the axes
    plt.title(('Predicted', predictions[i], 'real', y_test[i]))
    plt.imshow(image, cmap = 'gray')
    i += 1