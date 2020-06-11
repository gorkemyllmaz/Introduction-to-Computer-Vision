"""
Author: Görkem Yılmaz
Date: March 25, 2020
Introduction to Computer Vision 
Homework 2
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import h5py
import numpy as np
import matplotlib.pyplot as plt 

# PART 1 AND 2

# Read h5 file
test = h5py.File('test_catvnoncat.h5', 'r')
test_set_x = test['test_set_x']
test_set_y = test['test_set_y']
train = h5py.File('train_catvnoncat.h5', 'r')
train_set_x = train['train_set_x']
train_set_y = train['train_set_y'] 

train_set_y = np.reshape(train_set_y, (1,train_set_y.shape[0]))
test_set_y = np.reshape(test_set_y, (1,test_set_y.shape[0]))

train_set_x_flatten = np.reshape(train_set_x, (train_set_x.shape[0], -1)).T
test_set_x_flatten = np.reshape(test_set_x,(test_set_x.shape[0], -1)).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# Initialize W, b, learning rate, m, epochs
W = np.zeros((train_set_x.shape[0],1))
b = 0
learningRate = 0.0001
m = 50
epochs = 50

# Implementation of the second pseudo-algorithm
def second(X,W,b,Y,learningRate,m,epochs):
    costList = []
    for i in range(epochs):
        Z = np.dot(W.T, X) + b
        A = 1 / (1 + (np.exp(-Z)))
        cost = (-1. / m) * np.sum((Y*np.log(A) + (1 - Y)*np.log(1-A)), axis=1)
        costList.append(cost[0])
        dZ = A-Y
        dW = (1/m)*np.dot(X,(dZ.T))
        db = (1/m)*np.sum(dZ)

        W = W - learningRate * dW
        b = b - learningRate * db
    return W, b, costList

# Implementation of the first pseudo-algorithm
def first(X,W,b,Y,learningRate,m,epochs):
    J = 0
    dW = 0
    db = 0
    costList = []
    for i in range(epochs):
        for i in range(m):
            Z = np.dot(W.T, X) + b
            A = 1 / (1 + (np.exp(-Z)))
            J += -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
            dZ = A - Y
            dW = dW + np.dot(X,dZ.T)
            db = db + np.sum(dZ)
        J = J / m
        costList.append(J)
        dW = dW /m
        db = db/m
        W = W - learningRate*dW
        b = b - learningRate*db
    return W,b,costList

# Implementation of the Binary Classifier with Logistic Regression
def binaryClassifier(rgbimage, W, b):
    prediction = np.zeros((1,rgbimage.shape[1]))
    Z = np.dot(W.T, rgbimage) + b
    A = 1 / (1 + np.exp(-Z))
    
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            prediction[0, i] = 1  
        else:
            prediction[0, i] = 0
    return prediction

# Getting updated weight, bias and cost values.
wFinal, bFinal, costFinal = second(train_set_x, W, b, train_set_y, learningRate, m, epochs)
wFinal2, bFinal2, costFinal2 = second(test_set_x, W, b, test_set_y, learningRate, m, epochs)
# wFinal, bFinal, costFinal = first(train_set_x, W, b, train_set_y, learningRate, m, epochs)
y_prediction = binaryClassifier(test_set_x, wFinal, bFinal)
y_prediction_train = binaryClassifier(train_set_x, wFinal, bFinal)

# Calculating training and test accuracy
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - test_set_y)) * 100))

# Plotting the cost for both test and training data.
loss_train = costFinal
loss_val = costFinal2
epochs = range(1, 51)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Test loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# PART 3

# Read h5 file again
test = h5py.File('test_catvnoncat.h5', 'r')
test_set_x = test['test_set_x']
test_set_y = test['test_set_y']
train = h5py.File('train_catvnoncat.h5', 'r')
train_set_x = train['train_set_x']
train_set_y = train['train_set_y'] 

batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 64, 64

# the data, split between train and test sets
x_train = train_set_x
y_train = train_set_y
x_test = test_set_x
y_test = test_set_y

x_train = np.array(x_train)
x_test = np.array(x_test)
input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# SGD Algortihm.
# sgd = keras.optimizers.SGD(lr=0.001)
# model.compile(loss=keras.losses.categorical_crossentropy,
#                 optimizer=sgd,
#                 metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plotting loss for both test and train.
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, 31)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Test loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting accuracy for both test and train.
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, 31)
plt.plot(epochs, accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# For loop gives the error=> for epoch in range(initial_epoch, epochs):
# TypeError: 'range' object cannot be interpreted as an integer
# Therefore this function exists.
def differentLearnRate(learningRate,epochs):
    # sgd = keras.optimizers.SGD(learningRate)
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #                 optimizer=sgd,
    #                 metrics=['accuracy'])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1, 31)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Test loss')
    plt.title('Training and Validation Loss with Learning Rate: {}'.format(learningRate))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, 31)
    plt.plot(epochs, accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
    plt.title('Training and Validation Accuracy with Learning Rate: {}'.format(learningRate))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Plot with different learning rates.
differentLearnRate(0.1,30)
differentLearnRate(0.025,30)
differentLearnRate(0.005,30)
differentLearnRate(0.001,30)
differentLearnRate(0.0002,30)