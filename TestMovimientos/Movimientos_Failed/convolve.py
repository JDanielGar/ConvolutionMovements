# Junio 5 del 2018
# Observar las siguientes paginas:
# 1. https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html
# 2. https://www.youtube.com/watch?v=FmpDIaiMIeA
# (Las dos se complementan, observar cambios)
# (Opcional) 3. https://cambridgespark.com/content/tutorials/deep-learning-for-complete-beginners-recognising-handwritten-digits/index.html 

from recurrence import Recurrent_Photo, resize_images, show_image

from keras import backend
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

from sklearn.utils import shuffle

from time import sleep

import numpy as np

''' Take the data from enviroment '''

non_movement = Recurrent_Photo(50)
print('Prepare next movement...')
sleep(2)
movement = Recurrent_Photo(50)

''' Prepare the data to train '''

A = resize_images(non_movement.get_recurrence(), (32, 32))
A_labels = np.zeros(A.shape[0])

B = resize_images(movement.get_recurrence(), (32, 32))
B_labels = np.ones(B.shape[0])

X_train = np.concatenate((A, B), axis=0)
Y_train = np.concatenate((A_labels, B_labels), axis=0)
X_train, Y_train = shuffle(X_train, Y_train)

''' Prepare parameters of neural network '''

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons


num_train, height, width, depth = X_train.shape 
X_train = X_train.astype('float32') 
Y_train = Y_train.astype('float32')

''' Prepare layers of network '''

inp = Input(shape=(height, width, depth))

conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(1, activation='softmax')(drop_3)

''' Train model '''

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='binary_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

