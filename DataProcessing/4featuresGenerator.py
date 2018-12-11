#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:50:19 2018

@author: chizhang
"""
import cv2
import numpy as np

from os import path
from glob import glob  
import random

k = np.ones(shape=[3,3],dtype=np.uint8)


# get all files in a given folder path, by a given type of file
# dir_path: path directory 
# ext: the file type, eg: 'pdf'. Find all pdf files in this given folder
def getAllfiles(dir_path, ext):
    tempList = glob(path.join(dir_path,"*.{}".format(ext)))
    tempList += (glob(path.join(dir_path,"*.{}".format(ext.upper()))))
    return tempList

# image the given image from white to black
def inverse_color(image):
    height,width,temp = image.shape
    img2 = image.copy()
    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-image[i,j][0],255-image[i,j][1],255-image[i,j][2])
    return img2

# calculate number of adjacent rooms of this room
def adjacent(im,contours):
    
    met = np.zeros(shape=(len(contours),len(contours)))
    for i in range(len(contours)):
        for j in range(i+1,len(contours)):
            canvas = np.zeros(shape=(im.shape[0], im.shape[1], 1), dtype=np.uint8)
            cv2.drawContours(canvas,[contours[i]],0,(255,255,255),-1)
            cv2.drawContours(canvas, [contours[j]], 0, (255, 255, 255), -1)
            cv2.imshow("F0",canvas)
            canvas = cv2.dilate(canvas,k,iterations=20)
            im2, contours_, hierarchy = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_) == 1:
                met[i,j] = 1
                met[j,i] = 1
    return np.sum(met,axis=0).tolist()

# feature generator oxygen
def featuresGenerator_oxygen(iml, originalImg, numberFeatures, outPutcontourDir):
    TINY = 0.0001
    im = cv2.imread(iml)
    _, im = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY_INV)
    cv2.floodFill(im, None, (1, 1), (0, 0, 0), cv2.FLOODFILL_MASK_ONLY)
    im1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(im1, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    SA = 0
    SP = 0
    idList = []
    locationList = []
    areaList = []
    circuList = []
    densityList = []
    colorList = []
    first_Index = originalImg.index('/')+1
    idName = originalImg[originalImg.index('/',first_Index)+1: originalImg.index('.')]
    new_im = np.zeros(shape=(im.shape[0],im.shape[1],3))
    o_im = cv2.imread(originalImg)
    imCopy = o_im
    # conut adj rooms of this current contour
    contours_for_counting = []
    for c in contours:
        SA += float(cv2.contourArea(c))
        SP += float(cv2.arcLength(c, True))
    for c in contours:
        area = float(cv2.contourArea(c)) / SA * 100.0
        perimeter = float(cv2.arcLength(c, True)) / SP * 100.0
        circularity = perimeter * perimeter / (area + TINY)
        if ((area >= 0.0) and (area <= 1)):
            cv2.drawContours(im, [c], -1, (255, 255, 255), -1)
            continue
        else:
            B = random.randint(0, 200)
            G = random.randint(0, 200)
            R = random.randint(0, 200)
            m = cv2.moments(c)
            cx = int(m['m10'] / (m['m00'] + TINY))
            cy = int(m['m01'] / (m['m00'] + TINY))
            # print(area)
            cv2.drawContours(im, [c], -1, (B, G, R), 8)
            colorList.append((B, G, R))
            locationList.append((cx, cy))
            areaList.append(area)
            circuList.append(circularity)
            # add density for each contour
            new_im = np.zeros(shape=(im.shape[0], im.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(new_im,[c],0,(255,255,255),-1)
            new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
            _,new_im = cv2.threshold(new_im,127,255,cv2.THRESH_BINARY)
            o_i = o_im.copy()
            o_i = cv2.cvtColor(o_i,cv2.COLOR_RGB2GRAY)
            _,o_i = cv2.threshold(o_i,220,255,cv2.THRESH_BINARY)
            o_i[new_im!=255] = 255
            densityList.append(np.sum(o_i==0)/np.sum(new_im))
            contours_for_counting.append(c)
    # add number of adjacent room for each contour
    adjacentRoom = adjacent(o_im, contours_for_counting)
    for i in range(len(locationList)):
        idList.append(idName + '_room' + str(i + 1))
        cv2.putText(imCopy, "Room" + str(i + 1), (locationList[i][0] - 30, locationList[i][1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colorList[i], 2)
        cv2.putText(imCopy, str(round(areaList[i], 2)), (locationList[i][0] - 30, locationList[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i], 2)
        cv2.putText(imCopy, str(round(circuList[i], 2)), (locationList[i][0] - 30, locationList[i][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i], 2)        

    featuresList = [idList, areaList, circuList, densityList, adjacentRoom]
    features = np.array(featuresList).T
    outPut = outPutcontourDir
    cv2.imwrite(outPut + 'contour_' + idName +'.jpg', imCopy)
    return features

# given the contour image and its corresponding original image
# input: 
# @contourImg: 'segmentation-EKO/Images/lot_plan_01A001.jpg' 
# @originalImg: 'segmentation-EKO/segmentedImages/lot_plan_01A001.jpg'
# @outPutcontourDir: '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/Tests/EkoContourLabel/'
# output: the info of each room in this floorplan image as a np arrary:
# @ id:
# @ relativeArea:
# @ density:
# @ compactness: 
def featuresGenerator_eko(contourImg, originalImg, numberFeatures, outPutcontourDir):
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
    compactnessList = []
    densityList = []
    numberRooms = 1
    first_Index = originalImg.index('/')+1
    idName = originalImg[originalImg.index('/',first_Index)+1: originalImg.index('.')]
    new_im = np.zeros(shape=(im.shape[0],im.shape[1],3))
    o_im = cv2.imread(originalImg)
    imCopy = o_im
    contours_for_counting = []
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
            # save all contours
            contours_for_counting.append(c)
            # add id for each contour
            idList.append(idName+ '_room' +str(numberRooms))
            # add area for each contour
            areaList.append(area)
            perimeter = float(cv2.arcLength(c, True)) / SP * 100.0
            compactness = (perimeter ** 2) / area
            compactnessList.append(compactness)
            # add color for each contour
            colorList.append((B, G, R))
            # add density for each contour
            new_im = np.zeros(shape=(im.shape[0], im.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(new_im,[c],0,(255,255,255),-1)
            new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
            _,new_im = cv2.threshold(new_im,127,255,cv2.THRESH_BINARY)
            o_i = o_im.copy()
            o_i = cv2.cvtColor(o_i,cv2.COLOR_RGB2GRAY)
            _,o_i = cv2.threshold(o_i,220,255,cv2.THRESH_BINARY)
            o_i[new_im!=255] = 255
            densityList.append(np.sum(o_i==0)/np.sum(new_im))
            # get each contour enter
            m = cv2.moments(c)
            cx = int(m['m10'] / (m['m00'] + TINY))
            cy = int(m['m01'] / (m['m00'] + TINY))
            locationList.append((cx, cy))
            numberRooms +=1
    adjacentRoom = adjacent(o_im, contours_for_counting)
    outPut = outPutcontourDir
    # print room number on each contour
    for i in range(len(areaList)):
        cv2.putText(imCopy, 'room: '+ str(i + 1), (locationList[i][0] - 30, locationList[i][1] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, colorList[i], 3)
        cv2.putText(imCopy, str(round(areaList[i], 2)), (locationList[i][0] - 30, locationList[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, colorList[i], 3)
        cv2.putText(imCopy, str(round(compactnessList[i], 2)), (locationList[i][0] - 30, locationList[i][1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, colorList[i], 3)
    print('saving image with label: '+ idName)
    cv2.imwrite(outPut + 'contour_' + idName+'.jpg', imCopy)
    featuresList = [idList, areaList, compactnessList, densityList, adjacentRoom]
    features = np.array(featuresList).T
    return features

# =============================================================================
# ori_img = 'segmentation-EQUATION/Images/lot_plan_01A101.jpg'
# con_img = 'segmentation-EQUATION/segmentedImages/lot_plan_01A101.jpg'
# eko_outputDir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/segmentation-EQUATION/ImagesLabel/' 
# f1 = featuresGenerator_oxygen(con_img, ori_img,4, eko_outputDir)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# =============================================================================


# given original, and contour folders, output the info of each room in each floorplan image
#input: @origin_dir = 'Eko/', @contour_dir = 'EkoContour/', @floorType: type of the floor
# output the info of each room as a pd datafram and save as a csv file
# titles: @id, @relativeArea, @density, @compactness
def featuresGeneratorAllFiles(origin_dir, contour_dir, numberFeatures, outPutcontourDir, floorType):
    allRoomsFeatures = np.empty((0, numberFeatures))
    # get original and contour images as lists
    originImgs = getAllfiles(origin_dir, 'jpg')
    originImgs.sort()
    contourImgs = getAllfiles(contour_dir, 'jpg')
    contourImgs.sort()
    
    for orgImgIndex, orgImgFile in enumerate(originImgs):
        print(orgImgFile)
        print(contourImgs[orgImgIndex])
        if(floorType == 'oxygen' or floorType =='equation'):
            currRoomsFeatures = featuresGenerator_oxygen(contourImgs[orgImgIndex], orgImgFile, numberFeatures, outPutcontourDir)
        else:
            currRoomsFeatures = featuresGenerator_eko(contourImgs[orgImgIndex], orgImgFile, numberFeatures, outPutcontourDir)
            
        allRoomsFeatures = np.vstack((allRoomsFeatures, currRoomsFeatures))
        
    # save allRoomsFeatures as a csv file
    floorPlanName = origin_dir[origin_dir.index('-')+1:origin_dir.index('/')]   
    np.savetxt(floorPlanName + '_w4_features.csv', allRoomsFeatures, delimiter=',', fmt='%s')
    return allRoomsFeatures


# run this featuresGenerator under dir: '/MachineLearningProject/DataProcessing' 
# get all feature samples for eko floorplan
eko_outputDir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/segmentation-EKO/ImagesLabel/'
ekoImageDir = 'segmentation-EKO/Images'
ekoContourDir = 'segmentation-EKO/segmentedImages/'
featuresGeneratorAllFiles(ekoImageDir, ekoContourDir, 5, eko_outputDir, 'eko')     


# get all feature samples for OXYGEN floorplan
oxy_outputDir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/segmentation-OXYGEN/ImagesLabel/'
oxyImageDir = 'segmentation-OXYGEN/Images'
oxyContourDir = 'segmentation-OXYGEN/segmentedImages/'
featuresGeneratorAllFiles(oxyImageDir, oxyContourDir, 5, oxy_outputDir, 'oxygen')     
  
# get all feature samples for EQUATION floorplan
eq_outputDir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/segmentation-EQUATION/ImagesLabel/'
eqImageDir = 'segmentation-EQUATION/Images'
eqContourDir = 'segmentation-EQUATION/segmentedImages/'
featuresGeneratorAllFiles(eqImageDir, eqContourDir, 5, eq_outputDir, 'eko')     
  
        
    
    
    
    
    
