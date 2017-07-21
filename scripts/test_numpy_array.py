import numpy as np
import cv2 as cv

input_data = np.load('/home/kohpai/Projects/Pedestrian Detection/data/Daimler Dataset-48x96(15500-Gray-pgm)/DaimlerBenchmark/Data/TrainingData/NonPedestrians/non_pedestrian-48x96x3-dataset.npy')

#  print('data shape: ', input_data[0][0].shape)
for i in range(20):
    print(input_data[i][1])
    cv.imshow("image: " + str(i), input_data[i][0].reshape(96, 48, 3))
    cv.waitKey(0)
