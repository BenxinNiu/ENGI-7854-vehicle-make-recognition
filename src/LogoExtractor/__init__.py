# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import cv2

# TODO :::: fucking remove this global var. Who the fuck would put this as global var??? silly B
plate=[]

def HSVfilter(source):
    imgHSV=cv2.cvtColor(source,cv2.COLOR_BGR2HSV)
    lowerBound=np.array([100,100,80])
    upperBound=np.array([130,255,255])
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    return mask
def apply(source):
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(source, element2, iterations = 1)
    img = cv2.erode(img, element1, iterations = 3)
    img = cv2.dilate(img, element2, iterations = 3)
    return img

def process(source):
    source=cv2.medianBlur(source,5)
    return apply(source)

def plateDetect(img,img2):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        x,y,w,h=cv2.boundingRect(con)
        area, ratio =w*h, w/h
        if ratio>2 and ratio<4 and area>=2000 and area<=25000:
            logo_y1, logo_y2, logo_x1, logo_x2 =max(0,int(y-h*3.0)),y,x,x+w
            img_logo=img2.copy()
            logo=img_logo[logo_y1:logo_y2,logo_x1:logo_x2]
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img2,(logo_x1,logo_y1),(logo_x2,logo_y2),(0,255,0),2)
            plate=[x,y,w,h]
            return logo, plate

def logoDetect(rough_area,image,plate):
    img=cv2.cvtColor(rough_area,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(2*img.shape[1], 2*img.shape[0]),interpolation=cv2.INTER_CUBIC)
    ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=cv2.Canny(img, 100,200)
    img = apply(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sigma=0
    result=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        area,ratio = w*h,max(w/h,h/w)
        if area>400 and area<20000 and ratio<2 and area>sigma:
                sigma=area
                result=[x,y,w,h]
                ratio2=ratio
    logo_X=[int(result[0]/2+plate[0]-3),int(result[0]/2+plate[0]+result[2]/2+3)]
    logo_Y=[int(result[1]/2+max(0,plate[1]-plate[3]*3.0)-3),int(result[1]/2+max(0,plate[1]-plate[3]*3.0)+result[3]/2)+3]
    logo=image[logo_Y[0]:logo_Y[1],logo_X[0]:logo_X[1]]
    return logo

def logoExtraction(srcPath):
    img = cv2.imread(srcPath)
    plateImg = HSVfilter(img)
    plateImg = process(plateImg)
    rough_area, plate = plateDetect(plateImg, img)
    return logoDetect(rough_area, img,plate)
