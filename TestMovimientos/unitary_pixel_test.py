import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep

camera = cv2.VideoCapture(0)
previous_photos = 0
X = []
N = []
for vc in range(100):
    _, photo = camera.read()
    cv2.imshow('Image', photo)

    # Point of PHOTO
    point = photo[0, 0, 0]

    # Add X to plot
    X.append(point)

    # Normalization
    N.append(np.sum(X)/len(X))
    m = 10
    
    # Printing Data
    print(photo[0, 0, 0])
    cv2.waitKey(1)
plt.plot(range(100), comportament)
plt.show()
plt.plot(range(100), normalization)
plt.show()