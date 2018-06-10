# Junio 5 del 2018
# Observar las siguientes paginas:
# 1. https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html
# 2. https://www.youtube.com/watch?v=FmpDIaiMIeA
# (Las dos se complementan, observar cambios)
# (Opcional) 3. https://cambridgespark.com/content/tutorials/deep-learning-for-complete-beginners-recognising-handwritten-digits/index.html 

from recurrence import Recurrent_Photo, resize_images, show_image

from keras import backend
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from sklearn.utils import shuffle

from time import sleep

import numpy as np

''' Take the data from enviroment '''

left = Recurrent_Photo(100).recurrence_image
right = Recurrent_Photo(100).recurrence_image

''' Prepare the data to train '''

A = resize_images(left, (100, 100))
A_labels = np.zeros(A.shape[0])

B = resize_images(right, (100, 100))
B_labels = np.ones(B.shape[0])

X_train = np.concatenate((A, B), axis=0)
Y_train = np.concatenate((A_labels, B_labels), axis=0)
X_train, Y_train = shuffle(X_train, Y_train)


''' Prepare layers of network '''


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Conv2D(58, (3, 3), activation = 'relu'))
classifier.add(Conv2D(58, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


''' Compiling classificator '''

batch_size = 32 
num_epochs = 200 

classifier.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

classifier.fit(X_train, Y_train,              
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1)

