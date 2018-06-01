import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep

    
intervals = 1
class Momentum():
    def __init__(self, dimensions, capacity, humbral=200):
        '''
        Dimensions is the number of sequences that you will have in
        your robot.
        The capacity is the time of sequence image processing.
        '''
        self.humbral = humbral
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
        print(self.momentum)
        self.capacity = capacity
        self.get_data(dimensions, capacity)
    def get_data(self, dimensions, capacity):
        self.capture = cv2.VideoCapture(0)
        self.last_sequence = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        for dimension in range(dimensions):
            for frame in range(capacity):
                ret, image = self.capture.read()
                self.momentum[dimension, frame] = image
                if len(self.last_sequence) == 0:
                    self.last_sequence.append(image)
                    cv2.putText(image,str(frame),(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                    cv2.imshow('frame',image)
                else:
                    mean = abs(image - self.last_sequence[-1])
                    self.last_sequence.append(image)
                    image[mean[:, :] >= self.humbral] = 0 
                    cv2.putText(image,str(frame),(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                    cv2.imshow('frame',image)
                    cv2.waitKey(1)
            print('Dimension', self.tags[dimension], ' completed.')
        self.capture.release()
        cv2.destroyAllWindows()
    def show_sequence(self, dimension):
        for capture in self.momentum[dimension]:
            cv2.imshow(self.tags[dimension-1], capture)
            cv2.waitKey(1)
    def train():
        # Parte de prediccion pendiente.
        pass

memory = Momentum(2, 100, 200)
memory.show_sequence(0)

# memory.show_last_sequence()


# Una red neuronal para cada canal de color
# def get_data():
#     '''
#     Get photo data.
#     '''

#     cap = cv2.VideoCapture(0)
#     moments = []
#     counter = 0

#     while(counter<35):
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # Display the resulting frame
#         cv2.imshow('frame',frame)

#         moments.append(frame)
        
#         counter+=1
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
#     return moments

# # Implementar una red neuronal convolucional aqui.
# # No vamos a coger unicamente los canales de blanco y negro, sino de todos los colores.
# # El array es de la forma: R, B, G, A.
# # Tratamos 4 dimensiones, en cada pixel.


# # Canal Rojo, Azul y Verde entran en una maquina.
# # La maquina compara una la imagen de su interior y la del nuevo momento.
# # La maquina muda su momento.

# class Momentum:

#     def __init__(self, record_array):
#         '''
#         First moma
#         '''
#         self.record_array = record_array

#     def convolucion(self):
#         convolution_image = []
#         for momentum in range(len(self.record_array)):
#             convolution_image.append(momentum)

            

#     def get_alfa(self, R, B, G):
#         self.red = self.red - R  
#     def get_delta(self, R, B, G):
#         # Suma de los 3 canales para mejor comprension. Solo una teoria.
#         R + B
    
#     # Definir el humbral automaticamente. 