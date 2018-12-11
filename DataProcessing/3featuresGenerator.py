#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:50:19 2018

@author: chizhang
"""
import cv2
import numpy as np
import pandas as pd
import csv
from os import path
from glob import glob  
import random

# get all files in a given folder path, by a given type of file
# dir_path: path directory 
# ext: the file type, eg: 'pdf'. Find all pdf files in this given folder
def getAllfiles(dir_path, ext):
    tempList = glob(path.join(dir_path,"*.{}".format(ext)))
    tempList += (glob(path.join(dir_path,"*.{}".format(ext.upper()))))
    return tempList

# given the contour image and its corresponding original image
# input: @contourImg: 'EkoContour/lot_plan_01A001.jpg' @originalImg: 'Eko/lot_plan_01A001.jpg'
# output: the info of each room in this floorplan image as a np arrary:
# TODO
# @ id:
# @ relativeArea:
# @ density:
# @ circularity: 
def featuresGenerator(contourImg, originalImg, numberFeatures):
    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    TINY = 0.0001
    im = cv2.imread(contourImg)
    _, im = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)
    cv2.floodFill(im, None, (1, 1), (0, 0, 0), cv2.FLOODFILL_MASK_ONLY)
    im1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im1 = cv2.erode(im1, kernel1, iterations=5)
    im1 = cv2.dilate(im1, kernel2, iterations=5)
    _, thresh = cv2.threshold(im1, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    SA = 0
    SP = 0 # sum of perimeter
    locationList = []
    areaList = []
    idList = []
    colorList = []
    features = []
    circularityList = []
    densityList = []
    numberRooms = 1
    idName = originalImg[originalImg.index('/')+1: originalImg.index('.')]
    new_im = np.zeros(shape=(im.shape[0],im.shape[1],3))
    density = []

    #calculate sum of the contour areas
    for c in contours:
        SA += float(cv2.contourArea(c))
        SP += float(cv2.arcLength(c, True))
    for c in contours:
        #find relaitveArea of each contour
        area = float(cv2.contourArea(c)) / SA * 100.0
        # if the contour is larger than 1.0 we find the info of this contour
        if (area >= 1.0):
            B = random.randint(0, 200)
            G = random.randint(0, 200)
            R = random.randint(0, 200)
            idList.append(idName+ '_room' +str(numberRooms))
            areaList.append(area)
            perimeter = float(cv2.arcLength(c, True)) / SP * 100.0
            circularity = (perimeter ** 2) / area
            circularityList.append(circularity)
            # adding color for each contour
            colorList.append((B, G, R))
            
            cv2.drawContours(new_im,[c],0,(255,255,255),-1)
            new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
            _,new_im = cv2.threshold(new_im,127,255,cv2.THRESH_BINARY)
            o_i = originalImg.copy()
            o_i[new_im!=255] = 0
            density.append(np.sum(o_i))
            # get each contour enter
            m = cv2.moments(c)
            cx = int(m['m10'] / (m['m00'] + TINY))
            cy = int(m['m01'] / (m['m00'] + TINY))
            locationList.append((cx, cy))
            numberRooms +=1
    
    outPut = '/EkoContourLabel/'
    # print room number on each contour
    for i in range(len(areaList)):
        cv2.putText(im, str(numberRooms), (locationList[i][0] - 30, locationList[i][1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colorList[i], 2)
        cv2.putText(im, str(round(areaList[i], 2)), (locationList[i][0] - 30, locationList[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i], 2)
        cv2.putText(im, str(round(circularityList[i], 2)), (locationList[i][0] - 30, locationList[i][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i], 2)
    cv2.imwrite(outPut + "contour_" + idName+'.jpg', im)
    print(idList)
    print(areaList)
    print(circularityList)
    print(densityList)
    return features


contourImg = 'EkoContour/lot_plan_01A001_0001.jpg'
originalImg =  'Eko/lot_plan_01A001.jpg'
# given original, and contour folders, output the info of each room in each floorplan image
#input: @origin_dir = 'Eko/', @contour_dir = 'EkoContour/'
# output the info of each room as a pd datafram and save as a csv file
# titles: @id, @relativeArea, @density, @circularity
def featuresGeneratorAllFiles(origin_dir, contour_dir, numberFeatures):
    allRoomsFeatures = []
    # get original and contour images as lists
    originImgs = getAllfiles(origin_dir, 'jpg')
    originImgs.sort()
    contourImgs = getAllfiles(contour_dir, 'jpg')
    contourImgs.sort()
    for orgImgIndex, orgImgFile in enumerate(originImgs):
        currRoomsFeatures = featuresGenerator(contourImgs[orgImgIndex], orgImgFile, numberFeatures)
        allRoomsFeatures += currRoomsFeatures
        
    # save allRoomsFeatures as a csv file
    floorPlanName = origin_dir[:origin_dir.index('/')]   
    np.savetxt(floorPlanName + '_features.csv',allRoomsFeatures, delimiter=',', fmt='%s')
    return allRoomsFeatures
        
        
    
    
    
    
    
