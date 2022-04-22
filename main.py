#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split


# # Identity block
def identity_block(X, filters):
    
    X_shortcut = X
    
    # Main path
    X = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)

    # Add main and shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


# # Convolution Block
def convolutional_block(X, filters, s=2):

    X_shortcut = X

    # MAIN PATH  
    X = Conv2D(filters=filters, kernel_size=(3, 3), strides=(s, s), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)


    X = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)

    # SHORTCUT
    X_shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # Add the Main and short path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def resNet(input_shape = (64, 64, 3), classes = 6):
    
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, filters = 64)
    X = identity_block(X, 64)

    # Stage 3
    X = convolutional_block(X, filters=128)
    X = identity_block(X, 128)

    # Stage 4
    X = convolutional_block(X, filters=256)
    X = identity_block(X, 256)

    # Stage 5
    X = convolutional_block(X, filters=512)
    X = identity_block(X, 512)

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model



def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
model = resNet(input_shape = (32, 32, 3), classes = 10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
a = load_cfar10_batch('cifar-10',1)
ta = a[0]
labels = a[1]

# Normalise the data
data = data/255.


# One hot encode the labels
encoded = np.zeros((10000,10))
for i,v in enumerate(labels):
    encoded[i][v] = 1


X_train, X_test, Y_train, Y_test = train_test_split(data, encoded, test_size = 0.2)


# # Model Fitting
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

# Prediction
preds = model.evaluate(X_test, Y_test)

