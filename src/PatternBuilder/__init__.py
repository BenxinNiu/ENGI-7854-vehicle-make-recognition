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

    def buildAll(self, saveCache=False):
        if self.cache.loadHistogramData():
            return
        print("start building local binary patterns \n"),
        for brand in self.knownBrandList:
            logoPath = os.path.join(self.sourcePath, brand)
            if os.path.isdir(logoPath):
                # keep track of the id for each car brand
                self.cache.brand.append(brand)
                self.buildFromLogoFiles(logoPath)
            else:
                print("Unable to find training data for {}, Skipping it \n".format(brand))
                print("Finished Building... To save the traning data, pass true to buildAll method")
        if saveCache:
            self.cache.saveHistogramData()
            print("finished building LBP.. Saving it to file: {}\n".format(self.cache.cacheSource))

    def buildFromLogoFiles(self, logoPath):
        histogram_list = []
        print("building for {}\n".format(logoPath))
        for file in os.listdir(logoPath):
            sourceImg = "{}/{}".format(logoPath, file)
            histogram, lbp = Util.computeLBPHistogram(cv2.imread(sourceImg), self.params)
            histogram_list.append(histogram)
        self.cache.histogram.append(np.mean(histogram_list, axis=0))
        print("finished building for {}\n \n".format(logoPath))
