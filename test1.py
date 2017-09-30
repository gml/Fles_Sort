# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:41:24 2017

@author: Gavin
"""

import cv2
import glob
import numpy as np

CAP_WIDTH = 60 #mm

def nothing(c):
    pass

def imgShower(img_name,img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    
def findLevel(thresh):
    height,width = thresh.shape
    thresh_copy = np.zeros((height, width, 3), np.uint8)
    thresh_copy = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    biggest_cnts =sorted(contours, key=cv2.contourArea, reverse=True)
    biggest_cnts = biggest_cnts[:4]
    for cnts in biggest_cnts:
        x,y,w,h = cv2.boundingRect(cnts)
        if ((w > 8*h) and (w > .3*width)):
            level_cnt = cnts
            break
        if(w > .75*width):
            base_cnt = cnts
    imgShower("as",thresh_copy)

#    epsilon = 0.06*cv2.arcLength(base_cnt,True)
#    approx = cv2.approxPolyDP(base_cnt,epsilon,True)
    
    x,y,w,h = cv2.boundingRect(level_cnt)
    cv2.rectangle(thresh_copy,(x,y),(x+w,y+h),(0,255,0),2)
    
    x_b,y_b,w_b,h_b = cv2.boundingRect(base_cnt)   
    cv2.rectangle(thresh_copy,(x_b,y_b),(x_b+w_b,y_b+h_b),(0,255,255),2)
    
    imgShower("theone", thresh_copy)
    level_height = y+h
    base_height = int(y_b + (1/10*h_b))
    return level_height, base_height
        



    
def thresholding(itr,img):
    if (itr == 1):
        ret, thresh = cv2.threshold(img,135,255,cv2.THRESH_BINARY_INV)
    if (itr == 2):
        ret, thresh = cv2.threshold(img,25,255,cv2.THRESH_BINARY_INV)
        kernel = kernel = np.ones((15,15),dtype = np.uint16)
    if (itr == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        #blur = cv2.GaussianBlur(gray, (5,5),0)
#        #ret, thresh = cv2.threshold(blur,175,255,cv2.THRESH_BINARY_INV)
#        cv2.namedWindow('image')
#        cv2.createTrackbar('L','image',0,255,nothing)
#        cv2.createTrackbar('H','image',0,255,nothing)
#        while(1):
#            
#            L = cv2.getTrackbarPos('L', 'image')
#            H = cv2.getTrackbarPos('H', 'image')
#            ret, thresh = cv2.threshold(blur,L,H,cv2.THRESH_BINARY_INV)
#            kernel = kernel = np.ones((15,15),dtype = np.uint16)
#            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#            cv2.imshow("gray image",blur)
#            cv2.imshow("threshimg",closing)
#            k = cv2.waitKey(1) & 0xFF
#            if k == 27:   # hit escape to quit
#                break
#        cv2.destroyAllWindows()
        ret, thresh = cv2.threshold(gray,175,255,cv2.THRESH_BINARY_INV)
        kernel = kernel = np.ones((15,15),dtype = np.uint16)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh

    
def newImage(img,level):
    width = img.shape[1]
    new_img = img[level:base,0:width]
    imgShower("new img", new_img)
    return (new_img)
    
def calibration(img):
    global mmPP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9),0)
    thresh = thresholding(2,blur)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    imgShower("cap width", img)
    mmPP = CAP_WIDTH/w            #pixel to width ratio
    print("calibration complete with ptwr = ",mmPP)
    
def measureFunk(roi_thresh,img):
    global mmPP
    vol = 0
    imgShower("roi thresh", roi_thresh)
    left_values = []
    right_values = []
    height, width = roi_thresh.shape
#    print(height,width)
#    print(roi_thresh[0,])
    
#    print("values: ")
    for row in range(height):
        left_val = 0
        right_val = 0
        for col in range (width):
            l_val = col
            r_val = width - 1 - col
            if (roi_thresh[row][l_val] > 200 and left_val == 0):
                left_val = l_val
                left_values.append(left_val)
            if (roi_thresh[row][r_val] > 200 and right_val == 0):
                right_val = r_val
                right_values.append(right_val)
#        print(left_val, ", ",right_val)
        cv2.line(img,(left_val,row),(right_val,row),(0,0,255),5)
        radius = (right_val-left_val)/2
        area = np.pi * np.square(radius * mmPP)
        vol += (area * mmPP) * 0.001
    print ("bottle has a luquid volume of: ",int(vol), "ml")
#true vol 1 = 260, vol 2 = 340
        
    imgShower("blood lines", img)
 
                
    
    

        


def prepImg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vert = img.shape[0]                             #vertical size
    horz = img.shape[1]                             #horizontal size
    print(vert, ", ", horz)
    blurred_img = cv2.GaussianBlur(gray, (7,7),0)
    thresh = thresholding(1,blurred_img)
#    kernel = np.ones((50,20),np.uint8)
#    opening = cv2.morphologyEx(thresh[int(0.9*vert):], cv2.MORPH_OPEN, kernel)
#    thresh[int(0.9*vert):] = opening
    return thresh
    
    
i = 0
for img in glob.glob("images\*.jpg"):
    i += 1
    img =cv2.imread(img)
    res_img = cv2.resize(img,None, fx = .25, fy = .25)
    if (i == 1):
        calibration(res_img)
    imgShower("image",res_img)
    thresh = prepImg(res_img)
    level,base = findLevel(thresh)
    roi = newImage(res_img,level)
    roi_thresh = thresholding(3,roi)
    measureFunk(roi_thresh, roi)

    









