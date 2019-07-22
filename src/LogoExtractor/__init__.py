# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import cv2


# TODO :::: fucking remove this global var. Who the fuck would put this as global var??? silly B
plate=[]

# TODO ::: rewrite this packages

def HSVfilter(img):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(imgHSV)
    lowerBlue=np.array([100,100,80])
    upperBlue=np.array([130,255,255])
    mask=cv2.inRange(imgHSV,lowerBlue,upperBlue)
    plateImg=cv2.bitwise_and(img,img,mask=mask)
    return mask

def process(img):
    img=cv2.medianBlur(img,5)
    kernel=np.ones((3,3),np.uint8)
    sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 3)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(sobel, element2, iterations = 1)
    erosion = cv2.erode(dilation, element1, iterations = 1)
    dilation2 = cv2.dilate(erosion, element2,iterations = 3)
    #img=cv2.dilate(img,kernel,iterations = 1)
    #img=cv2.Canny(img,100,200)
    return dilation2

def plateDetect(img,img2):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        x,y,w,h=cv2.boundingRect(con)
        area=w*h
        ratio=w/h
        if ratio>2 and ratio<4 and area>=2000 and area<=25000:
            logo_y1=max(0,int(y-h*3.0))
            logo_y2=y
            logo_x1=x
            logo_x2=x+w
            img_logo=img2.copy()
            logo=img_logo[logo_y1:logo_y2,logo_x1:logo_x2]
            cv2.imwrite('./logo1.jpg',logo)
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img2,(logo_x1,logo_y1),(logo_x2,logo_y2),(0,255,0),2)
            global plate
            plate=[x,y,w,h]
            return logo


def logoDetect(img,imgo):
    imglogo=imgo.copy()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(2*img.shape[1],2*img.shape[0]),interpolation=cv2.INTER_CUBIC)
    ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=cv2.Canny(img,100,200)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img, element2, iterations = 1)
    img = cv2.erode(img, element1, iterations = 3)
    img = cv2.dilate(img, element2, iterations = 3)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    tema=0
    result=[]
    for con in contours:
        x,y,w,h=cv2.boundingRect(con)
        area=w*h
        ratio=max(w/h,h/w)
        if area>300 and area<20000 and ratio<2:
            if area>tema:
                tema=area
                result=[x,y,w,h]
                ratio2=ratio
    logo2_X=[int(result[0]/2+plate[0]-3),int(result[0]/2+plate[0]+result[2]/2+3)]
    logo2_Y=[int(result[1]/2+max(0,plate[1]-plate[3]*3.0)-3),int(result[1]/2+max(0,plate[1]-plate[3]*3.0)+result[3]/2)+3]
    cv2.rectangle(img,(result[0],result[1]),(result[0]+result[2],result[1]+result[3]),(255,0,0),2)
    cv2.rectangle(imgo,(logo2_X[0],logo2_Y[0]),(logo2_X[1],logo2_Y[1]),(0,0,255),2)
    print (tema,ratio2,result)
    logo2=imglogo[logo2_Y[0]:logo2_Y[1],logo2_X[0]:logo2_X[1]]

    return logo2


def logoExtraction(srcPath):
    img = cv2.imread(srcPath)
    plateImg = HSVfilter(img)
    plateImg = process(plateImg)
    logo = plateDetect(plateImg, img)
    logo2 = logoDetect(logo, img)
    return logo2
