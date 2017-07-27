import pyramid as pyr
import cv2 as cv

img = cv.imread('logs/img-0016-1.png')

for (i, tmp) in enumerate(pyr.getPyramid(img)):
    print(i)
    cv.imshow('test', tmp)
    cv.waitKey(0)
