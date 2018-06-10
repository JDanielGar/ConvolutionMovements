import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout

intervals = 1

class Momentum():
    def __init__(self, dimensions, capacity, humbral=255/2):
        '''
        Dimensions is the number of sequences that you will have in
        your robot.
        The capacity is the time of sequence image processing.
        '''
        print('Lets start with the tags.')
        sleep(intervals)
        print('You have:', dimensions, 'dimensions. Name it!')
        self.tags = []
        for number in range(dimensions):
            name = input('- {0} dimension. Her name is: '.format(number))
            self.tags.append(name)
            sleep(intervals)
            print('You call it', name)
            sleep(intervals)
        self.momentum = np.zeros([dimensions, capacity], dtype=object)
        self.humbral = humbral
        self.capacity = capacity
        self.get_data(dimensions, capacity)
        self.initialize_classifier()
    def initialize_classifier(self):
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape=(100, 100, 3), activation = 'relu'))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))
        classifier.add(Convolution2D(58, 3, 3, activation = 'relu'))
        classifier.add(Convolution2D(58, 3, 3, activation = 'relu'))
        classifier.add(MaxPool2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 512, activation = 'relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.classifier = classifier
    def get_data(self, dimensions, capacity):
        self.capture = cv2.VideoCapture(0)
        self.last_sequence = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        for dimension in range(dimensions):
            for frame in range(capacity):
                ret, image = self.capture.read()
                image = image/255 # Normalization
                if len(self.last_sequence) == 0:
                    self.momentum[dimension, frame] = image
                    self.last_sequence.append(image)
                    alfa_sequence = image # Sequence of transparency
                    cv2.imshow('frame',image)
                else:
                    self.last_sequence.append(image)
                    image = abs(image - self.last_sequence[-2]) # Compare with last momentum 
                    image[image[:, :] >= self.humbral] = 0 # Delete all points without minimum changes
                    alfa_sequence = image + alfa_sequence/1.5
                    self.momentum[dimension, frame] = alfa_sequence
                    cv2.imshow('frame', alfa_sequence)
                    cv2.waitKey(1)
            print('Dimension', self.tags[dimension], ' completed.')
            sleep(3)
        self.capture.release()
        cv2.destroyAllWindows()
    def show_sequence(self, dimension):
        for capture in self.momentum[dimension]:
            cv2.imshow(self.tags[dimension-1], capture)
            cv2.waitKey(1)
    def show_image(self, dimension, frame):
        cv2.imshow(self.tags[dimension-1], self.momentum[dimension, frame])
    def train_sequence(self):
        pass
    def generate_training_set(self):
        for dimension in range(len(self.momentum)):
            for photo in range(self.capacity):
                cv2.imwrite('./Data/{0}/{1}.png'.format(dimension, photo), self.momentum[dimension, photo]*255)   
        print('Operation Complete')             



        
memory = Momentum(2, 100)
memory.show_image(0, 5)
memory.generate_training_set()


# from keras.preprocessing.image import ImageDataGenerator

# classifier = memory.classifier

# train_datagen = ImageDataGenerator(
#         rescale=1,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)


# train_set = train_datagen.flow_from_directory(
#         'Data/',
#         target_size=(100, 100),
#         batch_size=32,
#         class_mode='binary')

# classifier.fit_generator(
#         train_set,
#         steps_per_epoch=100,
#         epochs=10
# )


# memory.show_sequence(0)

# Dividir en porcentajes para canales alfa:
# Ej: IMG1 * 0.25 + IMG2 * 0.75 ! WARN ESTO ESTA MAL
# La manera correctad e quitarle el brillo a un canal
# es dividiendolo por mas del 255. 
# 255 / 255 = 1
# 255/510 = 0.5 de opacidad

# La mejor forma
'''
Se normalizan todos los conales x/255
Al hacer la divicion se opacan
x = x/2 (primera)
y = y (segunda)
z = x+y (superposicion)
Por fin usaremos la proporcion aurea para la recurrencia
(El numero que permitira la opacidad sera ese)
h^2 = sqrt(a^2 + b^2)
'''