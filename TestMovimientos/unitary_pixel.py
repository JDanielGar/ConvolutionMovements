import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep

camera = cv2.VideoCapture(0)
previous_photos = 0
X = []
N = []
m = []
for vc in range(500):
    _, photo = camera.read()

    # Add X to plot
    X.append(photo)

    # Normalization
    norm = sum(X)/len(X)
    N.append(norm)
    if vc > 0:
        m.append(np.abs(norm - N[-2]))
        photo[m[-1] < 5] = 0
        cv2.imshow('Image', photo)
    
    cv2.waitKey(1)



def show(X):
    for photo in X:
        cv2.imshow('X', photo)
        cv2.waitKey(1)