from src.LocalBinaryPatternUtil.Params import Params
from src.PatternBuilder import PatternBuilder
from src.Classifier import  Classifier
import os

where = os.path.dirname(os.path.realpath(__file__))

logoIndex={'ChangAn':1,'DaZhong':2,'JiangHuai':3,'JinBei':4,'KaiRui':5,'QiRui':6,'XianDai':7}
sourcePath = where + "/src/resources/logo_template"
histogramPath = where + "/src/resources/lbph.dat"
params = Params(1, 6, 4, 4)

patternBuilder = PatternBuilder(params, logoIndex, sourcePath, histogramPath)
patternBuilder.buildAll()
patternBuilder.cache.saveHistogramData()

classifer = Classifier(patternBuilder.cache, params, logoIndex)
classifer.guessLogo(where+"/src/resources/logo2.jpg")
