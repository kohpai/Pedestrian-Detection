#! /usr/bin/python

import cv2 as cv
import sys

img_name = ""
out_height = 0
out_width = 0
out_name = ""

try:
    img_name = sys.argv[1]
    out_height = int(sys.argv[2])
    out_width = int(sys.argv[3])
    out_name = sys.argv[4]
except IndexError:
    print("Usage: central_crop $input $height $width $out_name")
    sys.exit()

img = cv.imread(img_name)
height, width, channels = img.shape
origin_y = int(height/2) - int(out_height/2)
dest_y   = origin_y + out_height
origin_x = int(width/2) - int(out_width/2)
dest_x   = origin_x + out_width

#  crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
crop_img = img[origin_y:dest_y, origin_x:dest_x]
#  cv.imshow(img_name[:img_name.index('.')], crop_img)
#  cv.waitKey(0)
cv.imwrite(
        img_name[:img_name.index('.')] +
        "-" +
        out_name +
        img_name[img_name.index('.'):], crop_img)
