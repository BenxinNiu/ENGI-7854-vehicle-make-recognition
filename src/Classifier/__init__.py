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

    def guessLogo(self, sourceImg):
        histogram, lbp = Util.computeLBPHistogram(sourceImg, self.params)
        minDiff = sys.float_info.max
        guess = -1
        for idx in range(len(self.cache.histogram)):
            diff = cv2.compareHist(self.cache.histogram[idx], histogram, cv2.HISTCMP_CHISQR)
            if diff < minDiff:
                minDiff = diff
                guess = self.cache.index[idx]
                print('label:%d distance:%f' % (guess, minDiff))
        # self.printGuess(guess, lbp, histogram)
        return guess, minDiff, lbp, histogram

    def printGuess(self, guess, sourceLBP, sourceHistogram, path):
        copy = self.knownBrandList
        new_label = {v: k for k, v in copy.items()}

        print('Guess:' + new_label[guess])

        plt.figure()
        # plt.subplot(121)
        # plt.imshow(sourceLBP, 'gray')
        #plt.title('LBP')
        # plt.subplot(122)
        plt.plot(sourceHistogram.flatten())
        plt.title('LBPH')
        plt.savefig(path+"lbph.png")
        plt.show()
        return new_label[guess]