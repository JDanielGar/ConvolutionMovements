# 05 de Junio del 2018 <- Se implementa en modulo interno
# Funcion para salvar imagenes ( Tutorial conv )

# 01 de Junio del 2018
# 31 Mayo 2018

import numpy as np
import matplotlib.pyplot as plt
from time import sleep

import cv2

# TODO Need to delete the normalization of the photos when the recurrence is making.

class Recurrent_Photo:
    '''
    Recurrent Photo only for testing
    '''
    def __init__(self, iterations=100, resize=(1280, 720)):
        self.camera = cv2.VideoCapture(0)
        self.video = np.zeros([iterations, resize[1], resize[0], 3])
        for iteration in range(iterations):
            self.video[iteration, :, :] = cv2.resize(
                (self.camera.read()[1]), 
                # FIXME: AHORA TRABAJAMOS CON LOS TRES CANALES
                resize
            )
            cv2.imshow('Prueba', self.video[iteration, :, :])
            cv2.waitKey(1)
        self.camera.release()
        self.resize = resize

    def get_recurrence(self, alpha=(0.55, 0.45)):
        '''
        Alpha are 2 float numbers represented by the amount of 
        superposition that you want to have in te current image.
        Example: 
            alpha = (0.5, 0.5) is neutral change, were the last
            image will have the same intensity of first image.
        '''
        first = np.array(self.video[0:self.video.shape[0]-1, :, :,])
        second = np.array(self.video[1:self.video.shape[0], :, :])
        diferences = self.get_diference(
            second, 
            first
        )
        # for image in range(len(diferences)):
        #     diferences[image] = diferences[image-1]* alpha[0] + diferences[image]* alpha[1]
        
        # Mirar ecuacion del cuaderno.
                
        return diferences

    def get_squared_recurrence(self):
        first_chanel = np.array(self.video[0:self.video.shape[0]-1], :, :])
        second_chanel = np.array(self.video[1:self.video.shape[0], :, :])
    
    def get_squared_diference(self, A, B):
        '''
        Get squared difference in photo, and return all those differences.
        '''
        return np.square(A - B)

    def get_diference(self, A, B):
        '''
        Get diference from two items
        '''
        absolute_error = np.abs(A - B)
        return absolute_error * (absolute_error >= 0)

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

test = Recurrent_Photo(10)
recurrence_test = test.get_recurrence()