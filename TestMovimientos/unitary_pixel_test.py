import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep

camera = cv2.VideoCapture(0)
previous_photos = 0
X = []
N = []
m = []
dx = []
for vc in range(100):
    _, photo = camera.read()
    cv2.imshow('Image', photo)

    # Point of PHOTO
    point = photo[0, 0, 0]

    # Add X to plot
    X.append(point)

    # Derivative
    if vc > 0:
        dx.append(X[-1]-X[-2])
    # Normalization
    # norm = np.sum(X)/len(X)
    # N.append(norm)
    # if vc > 0:
    #     m.append(np.abs(norm - N[-2]))
    
    # Printing Data
    print(photo[0, 0, 0])
    cv2.waitKey(1)
plt.plot(range(100), X)
plt.show()
plt.plot(range(99), dx)
plt.show()
# plt.plot(range(100), N)
# plt.show()
# plt.plot(range(99), m)
# plt.show()