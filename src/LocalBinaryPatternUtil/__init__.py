from __future__ import division  # 强制除法为浮点数
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings


def preprocess(img):
    '''pre-process image file by changing
     grey scale and
     apply median filter then
     perform historgram equalization
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    img = cv2.equalizeHist(img)
    return img


def computeLBPHistogram(sourceImgPath, params):
    resizedImg = cv2.resize(cv2.imread(sourceImgPath), (48, 48))
    localBinaryPattern = CircularLBP(resizedImg, params.radius, params.connectivity)
    histogram = LBPH(localBinaryPattern, int(math.pow(2, params.connectivity)), params.range_x, params.range_y)
    return histogram, localBinaryPattern

def CircularLBP(img, radius=1, neighbors=8):
    ''' compute circular Local Binary pattern '''
    src = img.copy()
    src = preprocess(src)
    if src.ndim == 3:
        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = src.shape
    dst = np.zeros((rows - 2 * radius, cols - 2 * radius), dtype=np.uint8)

    for k in range(neighbors):
        #
        rx = radius * math.cos(2.0 * math.pi * k / neighbors)
        ry = -radius * math.sin(2.0 * math.pi * k / neighbors)

        x1 = math.floor(rx)
        x2 = math.ceil(rx)
        y1 = math.floor(ry)
        y2 = math.ceil(ry)

        tx = rx - x1
        ty = ry - y1

        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        # iterate each pixels
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                # 获取中心像素点灰度值
                center = src.item(i, j)
                # formula
                neighbor = src.item(i + x1, j + y1) * w1 + \
                           src.item(i + x1, j + y2) * w2 + \
                           src.item(i + x2, j + y1) * w3 + \
                           src.item(i + x2, j + y2) * w4
                dst[i - radius, j - radius] |= (neighbor > center) << (neighbors - k - 1)
    return dst


def BlockLBPH(img, minValue, maxValue, normed=True):
    '''local LBP computation'''
    histSize = [maxValue - minValue + 1]
    ranges = [minValue, maxValue + 1]
    result = cv2.calcHist(img, [0], None, histSize, ranges)

    if normed:
        result = result / (int)(img.shape[0] * img.shape[1])
    return result.reshape(1, -1)


def LBPH(img, numPatterns, range_x, range_y, normed=True):
    src = img.copy()
    width = int(src.shape[1] / range_x)
    height = int(src.shape[0] / range_y)
    HistLBP = np.zeros((range_x * range_y, numPatterns), dtype=np.float32)
    if src.size == 0:
        return HistLBP.reshape(1, -1)

    cellIndex = 0
    for i in range(range_x):
        for j in range(range_y):
            src_cell = src[i * height:(i + 1) * height, j * width:(j + 1) * width]
            hist_cell = BlockLBPH(src_cell, 0, (numPatterns - 1), normed)
            HistLBP[cellIndex, :] = hist_cell
            cellIndex = cellIndex + 1
    return HistLBP.reshape(1, -1)
