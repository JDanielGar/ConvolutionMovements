# Junio 5 del 2018
# Observar las siguientes paginas:
# 1. https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html
# 2. https://www.youtube.com/watch?v=FmpDIaiMIeA
# (Las dos se complementan, observar cambios)
# (Opcional) 3. https://cambridgespark.com/content/tutorials/deep-learning-for-complete-beginners-recognising-handwritten-digits/index.html 

from recurrence import Recurrent_Photo

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

# Para continuar: 
Conv2D(32, (2, 2), input_shape=(100, 100, 3), activation='relu')
classifier.add(Convolution2D(32, 3, 3, , activation = 'relu'))


non_movement = Recurrent_Photo(50)

print('Prepare next movement...')
sleep(2)

movement = Recurrent_Photo(50)

X = resize_images(non_movement.get_recurrence())
Y = resize_images(movement.get_recurrence())

