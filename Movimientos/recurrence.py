import cv2

import numpy as np
import matplotlib.pyplot as plt

from time import sleep


class Recurrent_Photo():
    def __init__(self, capacity, humbral=255/2, times=1):
        capture = cv2.VideoCapture(0)

        self.last_sequence = []
        self.recurrence_image = np.zeros([capacity*times, 720, 1280, 3])

        image = capture.read()[1]/255
        
        self.last_sequence.append(image)

        alfa_sequence = image 

        for time in range(times):
            for counter in range(3):
                print(counter+1)
                sleep(1)
            for frame in range(capacity):
                # Capture image
                image = capture.read()[1]/255

                # Push image into array
                self.last_sequence.append(image)

                # Get diference
                image = np.abs(image - self.last_sequence[-2])

                # Get 0 when not change is detected
                image[image[:, :] >= humbral] = 0 

                # Alfa sequence
                alfa_sequence = image + alfa_sequence/1.5

                # Result into a numpy array
                self.recurrence_image[frame] = alfa_sequence

                # Normal CV2 Process
                cv2.imshow('Image', alfa_sequence)
                cv2.waitKey(1)
    
        capture.release()

def real_time_classification(times, classifier, humbral=255/2):
    '''
    Only works with keras classifier object.
    '''
    capture = cv2.VideoCapture(0)

    last_sequence = [0, 0]

    image = capture.read()[1]/255

    last_sequence[0] = image

    alfa_sequence = image 

    for times in range(times):
        # Capture image
        image = capture.read()[1]/255
        
        # Push image into array
        last_sequence.append(image)

        # Get diference
        image = np.abs(image - last_sequence[-2])

        # Get 0 when not change is detected
        image[image[:, :] >= humbral] = 0 

        # Alfa sequence
        alfa_sequence = image + alfa_sequence/1.5

        prediction = classifier.predict(np.array([resize_images(alfa_sequence)]))

        # Normal CV2 Process
        image_with_text = np.array(alfa_sequence)
        cv2.putText(image_with_text, 
            str(prediction),
            (100, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        cv2.imshow('Image', image_with_text)
        cv2.waitKey(1)

        last_sequence.pop(-2)

def resize_images(X, dimensions=(100, 100)):
    if len(X.shape) == 3:
        X = cv2.resize(X, dimensions)
    else:
        resized_images = np.zeros([len(X), dimensions[0], dimensions[1], 3])
        for image in range(len(X)):
            resized_images[image] = cv2.resize(X[image], dimensions)
        X = resized_images
    return X

def show_image(X):
    if len(X.shape) == 3:
        cv2.imshow('image', X)
    else:    
        for image in X:
            cv2.imshow('X', image)
            cv2.waitKey(1)
            sleep(0.05)

