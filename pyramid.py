import cv2 as cv
import numpy as np

def getPyramid(src, scale=2, minSize=(30, 30), skipSrc=True):
    if not skipSrc:
        yield src

    tmp = src.copy()

    while True:
        dst_width  = int(tmp.shape[1] / scale)
        dst_height = int(tmp.shape[0] / scale)

        if dst_width < minSize[1] or dst_height < minSize[0]:
            break

        tmp = cv.resize(tmp, dsize=(dst_width, dst_height), interpolation=cv.INTER_NEAREST)
        #  tmp = cv.pyrDown(tmp, dstsize=(dst_width, dst_height))

        yield tmp
