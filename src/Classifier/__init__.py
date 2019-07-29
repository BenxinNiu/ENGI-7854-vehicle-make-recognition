import cv2
import os
import sys
import src.LocalBinaryPatternUtil as Util
import matplotlib.pyplot as plt


class Classifier:

    def __init__(self, cache, params, knownBrandList):
        self.cache = cache
        self.params = params
        self.knownBrandList = knownBrandList

        self.lastGuess = ""
        self.lastMinDiff = 0
        self.lastLBP = []

    def guessLogo(self, sourceImg):
        histogram, self.lastLBP = Util.computeLBPHistogram(sourceImg, self.params)
        currentMin = cv2.compareHist(self.cache.histogram[0], histogram, cv2.HISTCMP_BHATTACHARYYA)
        print('brand:%s current min difference:%f' % (self.cache.brand[0], currentMin))
        guess = ""
        for idx in range(1, len(self.cache.histogram), 1):
            diff = cv2.compareHist(self.cache.histogram[idx], histogram, cv2.HISTCMP_BHATTACHARYYA)
            if currentMin > diff:
                currentMin = diff
                guess = self.cache.brand[idx]
                print('brand:%s current min difference:%f' % (guess, currentMin))

        if guess == "":
            guess=self.cache.brand[0]
        self.lastGuess = guess
        self.lastMinDiff = currentMin
        return guess, self.lastLBP


    def printGuess(self):
        if self.lastGuess == "":
            print("Something went wrong, unable to identify the brand")
        else:
            print("The vehicle brand is {}".format(self.lastGuess))
        # reset cache
        self.lastGuess = ""
        self.lastMinDiff = 0
        self.lastLBP = []
