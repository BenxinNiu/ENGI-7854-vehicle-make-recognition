from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def computeLBPHistogram(sourceImg, params):
    resizedImg = cv2.resize(sourceImg, (60, 60))
    lbp = computeLocalBinaryPattern(resizedImg, params.radius, params.connectivity)
    histogram = computeHistogram(lbp)
    return histogram, lbp


def computeLocalBinaryPattern(src, radius, neighbors):
    ''' compute Local Binary pattern '''
    """ algorithm source: https://www.bytefish.de/blog/local_binary_patterns/ """
    img = preprocess(src)
    dst = np.zeros((img.shape[0] - 2 * radius, img.shape[1] - 2 * radius), dtype=np.uint8)
    for n in range(neighbors):
        # sample points
        x = radius * math.cos(2.0 * math.pi * n / neighbors)
        y = -radius * math.sin(2.0 * math.pi * n / neighbors)

        #relative indices (offset from center)
        fx = math.floor(x)
        fy = math.ceil(x)
        cx = math.floor(y)
        cy = math.ceil(y)

        # fractional part
        tx = x - fx
        ty = y - cx

        # set interpolation weights
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        # iterate each pixels
        for i in range(radius, img.shape[0] - radius):
            for j in range(radius, img.shape[1] - radius):
                # formula
                neighbor = img[i + fx][j + cx] * w1 + img[i + fx][j + cy] * w2 + img[i + fy][j + cx] * w3 + img[i + fy][j + cy] * w4
                dst[i - radius][j - radius] = neighbor > img[i][j]
    return dst

def computeHistogram(lbp):
    return cv2.calcHist(lbp, [0], None, [256], (0, 256), accumulate=False)
