#! /usr/bin/python

import numpy as np
import tflearn
import sys
import os.path

model_name = None

try:
    model_name = sys.argv[1]
except IndexError:
    print("No model name specified")

data   = np.load('train_data-48x96.npy')
labels = np.load('train_labels-48x96.npy')

height = 96
width  = 48

# ------------------------------------------------------------------------------ #

#  input_layer = tflearn.input_data(shape=[None, height * width])
input_layer = tflearn.input_data(shape=[None, height, width, 1])

conv_pool = tflearn.conv_2d(input_layer, 8, [3, 3], activation='relu')
conv_pool = tflearn.max_pool_2d(conv_pool, [2, 2])
conv_pool = tflearn.conv_2d(input_layer, 16, [3, 3], activation='relu')
conv_pool = tflearn.max_pool_2d(conv_pool, [2, 2])

fc = tflearn.fully_connected(conv_pool, 32, activation='relu')
fc = tflearn.fully_connected(fc, 32, activation='relu')

output_layer = tflearn.fully_connected(fc, 2, activation='softmax')

cross_entropy = tflearn.regression(output_layer)

# ------------------------------------------------------------------------------ #

model = tflearn.DNN(cross_entropy, tensorboard_dir='logs')

if model_name is not None and os.path.isfile(model_name):
    print("Loading the saved weights from ", model_name)
    model.load(model_name)

try:
    #  model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
    model.fit(
            data.reshape([-1, height, width, 1]),
            labels,
            n_epoch=10,
            batch_size=100,
            show_metric=True,
            run_id=None)
except KeyboardInterrupt:
    print("Learning process was terminated")

if model_name is not None:
    print("Saving the weights to ", model_name)
    model.save(model_name)

# ------------------------------------------------------------------------------ #

#  dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
#  winslet = [1, 'Rose DeWitt Bukalet', 'female', 17, 1, 2, 'N/A', 100.0000]

#  dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)

#  pred = model.predict([dicaprio, winslet])

#  print("Dicaprio Surviving Rate: ", pred[0][1])
#  print("Dicaprio Dying Rate: ", pred[0][0])
#  print("Winslet Surviving Rate: ", pred[1][1])
#  print("Winslet Dying Rate: ", pred[1][0])
