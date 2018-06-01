
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout

# Create Classifier

classifier = Sequential()

# Add Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape=(100, 100, 3), activation = 'relu'))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))


# Pooling

classifier.add(MaxPool2D(pool_size = (2, 2)))

# DropOut

classifier.add(Dropout(0.25))

# Second Conv

classifier.add(Convolution2D(58, 3, 3, activation = 'relu'))
classifier.add(Convolution2D(58, 3, 3, activation = 'relu'))

classifier.add(MaxPool2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

# Flatten

classifier.add(Flatten())

# Full Conection

classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compile

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN:

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'Dataset/Training/',
        target_size=(100, 100),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Dataset/Test/',
        target_size=(100, 100),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=100,
        epochs=10,
        validation_data=test_set,
        validation_steps=20
)

# To Predict
 
total_data = []
while 100:
    total_data.append(get_data())

# Tomar imagen cada x tiempo
