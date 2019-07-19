from src.LocalBinaryPatternUtil.Params import Params
from src.PatternBuilder import PatternBuilder
from src.Classifier import  Classifier
import src.LogoExtractor as extractor
import os
import sys

where = os.path.dirname(os.path.realpath(__file__))

logoDictionary={'ChangAn':1,'VolksWagen':2, 'JiangHuai':3, 'JinBei':4, 'KaiRui':5, 'QiRui':6, 'Hyundai':7}
sourcePath = where + "/src/resources/logo_template"
histogramPath = where + "/src/resources/lbph.dat"
params = Params(1, 6, 4, 4)

identifyPic = sys.argv[1]
logo = extractor.logoExtraction(where+ identifyPic)

patternBuilder = PatternBuilder(params, logoDictionary, sourcePath, histogramPath)
patternBuilder.buildAll()
patternBuilder.cache.saveHistogramData()

classifer = Classifier(patternBuilder.cache, params, logoDictionary)
classifer.guessLogo(logo)
