import numpy as np
import matplotlib.pyplot as plt
from time import sleep

import cv2

class Recurrent_Photo:
    '''
    Recurrent Photo only for testing
    '''
    def __init__(self, iterations=100, resize=(1280, 720)):
        self.camera = cv2.VideoCapture(0)
        self.video = np.zeros([iterations, resize[1], resize[0]])
        for iteration in range(iterations):
            self.video[iteration, :, :] = cv2.resize(
                (self.camera.read()[1]/255)[:, :, 1], 
                resize
            )
            cv2.imshow('Prueba', self.video[iteration, :,])
            cv2.waitKey(1)
        self.backup = self.video
    def get_recurrence(self, threshold=0.5):
        first = np.array(self.video[0:self.video.shape[0]-1, :, :])
        second = np.array(self.video[1:self.video.shape[0], :, :])
        diferences = self.get_diference(
            second, 
            first
        )
        self.recurrence = second
        return diferences
        
    def get_diference(self, A, B):
        '''
        Get diference from two items
        '''
        return np.abs(A - B)
    
    def show_image(self, X):
        for image in X:
            cv2.imshow('X', image)
            cv2.waitKey(1)
            sleep(0.05)

        

data_marker = Recurrent_Photo(100)
X = data_marker.get_recurrence(0.1)

# TODO CANAL ALFA DE NUEVO
# TODO RECURRENCIA MAYORITARIA