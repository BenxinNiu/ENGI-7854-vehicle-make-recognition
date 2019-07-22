from __future__ import division
import math
import numpy as np
import cv2
import os
import src.LocalBinaryPatternUtil as Util
from src.PatternBuilder.InterimResult import InterimResult


class PatternBuilder:

    def __init__(self, params, knownBrandList, sourcePath, histogramPath):
        self.knownBrandList = knownBrandList
        self.sourcePath = sourcePath
        self.cache = InterimResult([], [], histogramPath)
        self.params = params

    def preprocessImg(self, file):
        try:
            return cv2.resize(cv2.imread(file), (48, 48))
        except OSError:
            print("\n Error occurred during pre-processing logo: {} \n".format(file))

    def buildAll(self):
        if self.cache.loadHistogramData():
            return
        print("start building local binary patterns \n"),
        for subDir in os.listdir(self.sourcePath):
            logoPath = os.path.join(self.sourcePath, subDir)
            if os.path.isdir(logoPath):
                # keep track of the id for each car brand
                self.cache.index.append(self.knownBrandList[subDir])
                self.buildFromLogoFiles(logoPath)
        print("finished building LBP.. Saving it to file: {}\n".format(self.cache.cacheSource))
        self.cache.saveHistogramData()

    def buildFromLogoFiles(self, logoPath):
        histogram_list = []
        print("building for {}\n".format(logoPath))
        for file in os.listdir(logoPath):
            sourceImg = "{}/{}".format(logoPath, file)
            histogram, lbp = Util.computeLBPHistogram(cv2.imread(sourceImg), self.params)
            histogram_list.append(histogram)
        self.cache.histogram.append(np.mean(histogram_list, axis=0))
        print("finished building for {}\n \n".format(logoPath))
