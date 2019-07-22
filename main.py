import sys
import os
import webbrowser

from src.LocalBinaryPatternUtil.Params import Params
from src.PatternBuilder import PatternBuilder
from src.Classifier import  Classifier
import src.LogoExtractor as extractor
import cv2


where = os.path.dirname(os.path.realpath(__file__))

logoDictionary={'ChangAn':1,'VolksWagen':2, 'JiangHuai':3, 'JinBei':4, 'KaiRui':5, 'QiRui':6, 'Hyundai':7}
sourcePath = where + "/src/resources/logo_template"
histogramPath = where + "/src/resources/lbph.dat"
params = Params(1, 6, 4, 4)

# TODO replace with testPic later ..
# testPic = sys.argv[1]
testPic = where + "/src/resources/testImages/highway_3.jpg"
# TODO replace with testPic later ..
logo = extractor.logoExtraction(testPic)

cv2.imwrite(where + '/output/logo.jpg', logo)

patternBuilder = PatternBuilder(params, logoDictionary, sourcePath, histogramPath)
patternBuilder.buildAll()
patternBuilder.cache.saveHistogramData()

classifer = Classifier(patternBuilder.cache, params, logoDictionary)
guessIdx, minDiff, lbp, histogram = classifer.guessLogo(logo)
cv2.imwrite(where + '/output/lbp.jpg', lbp)
guess = classifer.printGuess(guessIdx, lbp, histogram, where + "/output/")

report_template = open(where + "/src/resources/report_template.html", mode='r')
print("something")
# Hacky way to generate report.
# TODO use beautifulsoup, if time permits
html = report_template.read()
html = html.replace("......", guess)
html = html.replace("localBinaryPattern", where + '/output/lbp.jpg')
html = html.replace("histogram", where + '/output/lbph.png')
html = html.replace('extracted_logo', where + '/output/logo.jpg')
html = html.replace("source_img", testPic)
# close the file
report_template.close()
print (html)
report = open(where+"/output/report.html", "w")
report.write(html)
report.close()

print (html)

webbrowser.open('file://' + os.path.realpath(where+"/output/report.html"))