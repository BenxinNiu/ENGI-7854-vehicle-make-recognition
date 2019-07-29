import sys
import os
import webbrowser

from src.LocalBinaryPatternUtil.Params import Params
from src.PatternBuilder import PatternBuilder
from src.Classifier import  Classifier
from src.LogoExtractor import LogoExtractor
import cv2


where = os.path.dirname(os.path.realpath(__file__))

trainedLogoSet = ["VolksWagen", "Hyundai", "BMW"]
sourcePath = where + "/src/resources/logo_template"
histogramPath = where + "/src/resources/lbph.dat"
params = Params(1, 6, 4, 4)

# TODO replace with testPic later ..
# testPic = sys.argv[1]
testPic = where + "/src/resources/testImages/highway_5.jpg"
# testPic = where + "/src/resources/testImages/10.png"

extractor = LogoExtractor(testPic)
logo = extractor.getLogo()

cv2.imwrite(where + '/output/logo.jpg', logo)

patternBuilder = PatternBuilder(params, trainedLogoSet, sourcePath, histogramPath)
patternBuilder.buildAll(saveCache=True)

classifer = Classifier(patternBuilder.cache, params, trainedLogoSet)
guess, lbp = classifer.guessLogo(logo)
classifer.printGuess()
cv2.imwrite(where + '/output/lbp.jpg', lbp)


report_template = open(where + "/src/resources/report_template.html", mode='r')
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
report = open(where+"/output/report.html", "w")
report.write(html)
report.close()

webbrowser.open('file://' + os.path.realpath(where+"/output/report.html"))