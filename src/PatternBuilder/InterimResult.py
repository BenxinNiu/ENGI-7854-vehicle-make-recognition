import os
import pickle

class InterimResult:

    def __init__(self, histogram, index, cacheSource):
        self.histogram = histogram
        self.index = index
        self.cacheSource = cacheSource

    def saveHistogramData(self):
        try:
            pickle.dump([self.histogram, self.index], open(self.cacheSource, 'wb'))
        except OSError:
            print("Error occurred during saving file exiting: {}".format(self.cacheSource))

    def loadHistogramData(self):
        if os.path.exists(self.cacheSource):
            try:
                print("Loading data from: {}".format(self.cacheSource))
                self.histogram, self.index = pickle.load(open(self.cacheSource, 'rb'))
                return True
            except (ImportError, OSError) as ex:
                print("Error occurred during loading file exiting: {}".format(self.cacheSource))
        else:
            print("Could not load previously saved data due to file not found: {}\n \n ".format(self.cacheSource))
            return False
