import cv2 
import numpy as np
import matplotlib.pyplot as plt

from time import sleep

camera = cv2.VideoCapture(0)

sequence = []

for time in range(10):
    sequence.append(camera.read()[1])
    sleep(1)
    cv2.imshow('Hey', sequence[-1])
    cv2.waitKey(1)

camera.release()

W = sequence[1]
X = sequence[2]
Y = sequence[5]
Z = sequence[9]

A = np.abs(X - W)
B = np.abs(Y - X)
C = np.abs(Z - Y)

def get_data():
    # ASIGNACION CAMARA
    camera = cv2.VideoCapture(0)
    # PRIMER MOMENTO
    sequence = [camera.read()[1]]
    for frame in range(capacity):
        alfa_sequence = image
        image = np.abs(image - last_sequence[-2])
        image[image[:, :] >= self.humbral] = 0
        alfa_sequence = image + alfa_sequence/1.5
        cv2.imshow('frame', alfa_sequence)
        cv2.waitKey(1)
    print('Dimension', self.tags[dimension], ' completed.')
    sleep(3)
    self.capture.release()
    cv2.destroyAllWindows()
