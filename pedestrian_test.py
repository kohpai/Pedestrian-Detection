#! /usr/bin/python

import numpy as np
import tflearn
import sys
import os.path

model_name = None
data       = None
labels     = None
channels   = None
height     = None
width      = None

try:
    data       = np.load(sys.argv[1])
    labels     = np.load(sys.argv[2])
    height     = int(sys.argv[3])
    width      = int(sys.argv[4])
    channels   = int(sys.argv[5])
    model_name = sys.argv[6]

except IndexError:
    print("Usage: pedestrian_test $data.npy $labels.npy $height $width $channels $model_name")
    sys.exit()

# ------------------------------------------------------------------------------ #

#  input_layer = tflearn.input_data(shape=[None, height * width])
input_layer = tflearn.input_data(shape=[None, height, width, channels])

conv_pool = tflearn.conv_2d(input_layer, 32, [3, 3], activation='relu')
conv_pool = tflearn.max_pool_2d(conv_pool, [2, 2])
conv_pool = tflearn.conv_2d(input_layer, 32, [3, 3], activation='relu')
conv_pool = tflearn.max_pool_2d(conv_pool, [2, 2])

fc = tflearn.fully_connected(conv_pool, 32, activation='relu')
fc = tflearn.dropout(fc, 0.95)
fc = tflearn.fully_connected(fc, 32, activation='relu')
fc = tflearn.dropout(fc, 0.95)

output_layer = tflearn.fully_connected(fc, 2, activation='softmax')

cross_entropy = tflearn.regression(output_layer)

# ------------------------------------------------------------------------------ #

model = tflearn.DNN(cross_entropy, tensorboard_dir='logs')
model.load(model_name)

print("Evaluation score: ", model.evaluate(data.reshape([-1, height, width, channels]), labels))
