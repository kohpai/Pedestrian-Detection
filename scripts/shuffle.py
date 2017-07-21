#! /usr/bin/python

import numpy as np
import cv2 as cv
import sys

pos_data_name  = ""
neg_data_name  = ""
data_out_name  = ""
label_out_name = ""

try:
    pos_data_name  = sys.argv[1]
    neg_data_name  = sys.argv[2]
    data_out_name  = sys.argv[3]
    label_out_name = sys.argv[4]
except IndexError:
    print("Usage: shufdataset $pos $neg $data_out $label_out")
    sys.exit()

pos_data = np.load(pos_data_name)
neg_data = np.load(neg_data_name)
out_data = np.concatenate((pos_data, neg_data))

np.random.shuffle(out_data)

data_out  = [i[0] for i in out_data]
label_out = [i[1] for i in out_data]

data_out  = np.array(data_out)
label_out = np.array(label_out)

#  print(data_out.shape)
#  print(label_out.shape)

#  Test showing images
#  for i in range(20):
#      print(label_out[i])
#      cv.imshow("image: " + str(i), data_out[i].reshape(96, 48, 1))
#      #  print(out_data[i][1])
#      #  cv.imshow("image: " + str(i), out_data[i][0].reshape(96, 48, 1))
#      cv.waitKey(0)

np.save(data_out_name, data_out)
np.save(label_out_name, label_out)
