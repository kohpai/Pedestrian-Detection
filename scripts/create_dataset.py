#! /usr/bin/python

import numpy as np
import cv2 as cv
import sys
import os

img_dir = ""
height = 0
width = 0
channels = 0
class_vector = [0, 0]
out_name = ""

try:
    img_dir = sys.argv[1]
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    channels = int(sys.argv[4])
    class_vector = [1, 0] if int(sys.argv[5]) == 1 else [0, 1]
    out_name = sys.argv[6]
except IndexError:
    print("Usage: img2dataset $dir $height $width $channels $class $out_name")
    sys.exit()

data = []

for f in os.listdir(img_dir):
    img = cv.imread(os.path.join(img_dir, f))
    #  print(type(img)) # numpy.ndarray
    if img is not None:
        flattened_img = img.reshape(height * width, 3)
        if channels == 1:
            gray_img = []

            for i in flattened_img:
                gray_img.append(i[0])

            data.append([np.array(gray_img), np.array(class_vector)])
        else:
            data.append([flattened_img, np.array(class_vector)])

out_data = np.array(data)

#  Test size of images
#  for i in range(len(os.listdir(img_dir))):
#      print("pixels: %d, channels: %d" % (len(out_data[i][0]), len(out_data[i][1])))

#  Test showing images
#  for i in range(6):
#      cv.imshow('test', out_data[i][0].reshape(height, width, channels))
#      cv.waitKey(0)

np.save(out_name, out_data)
