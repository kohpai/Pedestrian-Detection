import pyramid as pyr
import cv2 as cv

img = cv.imread('img-0016-1.png')

for (i, tmp) in enumerate(pyr.getPyramid(img, skipSrc=False)):
    print(i)
    cv.imshow('test', tmp)
    cv.waitKey(0)
