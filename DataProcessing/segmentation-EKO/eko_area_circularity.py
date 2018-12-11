'''
    Author: Qitong Wang
    Date: 11/28/2018

    Description:
        Calculating each room's ralative area and circularity
'''

import cv2
import numpy as np
import random
import os

input = "/Users/wangqitong/PycharmProjects/542_test/eko_res/"
output = "/Users/wangqitong/PycharmProjects/542_test/eko_final_res/"

kernel1 = np.ones((5, 5), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

k = np.ones((3,3))
TINY = 0.0001



def cal_eachArea_eachCir(iml):
    im = cv2.imread(input + iml)
    _, im = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)
    cv2.floodFill(im, None, (1, 1), (0, 0, 0), cv2.FLOODFILL_MASK_ONLY)

    im1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im1 = cv2.erode(im1, kernel1, iterations=5)
    im1 = cv2.dilate(im1, kernel2, iterations=5)

    _, thresh = cv2.threshold(im1, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = 0
    SA = 0
    SP = 0
    locationList = []
    areaList = []
    circuList = []
    colorList = []

    for c in contours:
        SA += float(cv2.contourArea(c))
        SP += float(cv2.arcLength(c, True))

    for c in contours:
        area = float(cv2.contourArea(c)) / SA * 100.0
        perimeter = float(cv2.arcLength(c, True)) / SP * 100.0
        circularity = perimeter * perimeter / area
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
            print(area)
            cv2.drawContours(im, [c], -1, (B, G, R), 8)
            colorList.append((B, G, R))

            print(cx, cy)
            print(area)
            print(circularity)
            print()

            locationList.append((cx, cy))
            areaList.append(area)
            circuList.append(circularity)

    for i in range(len(locationList)):
        cv2.putText(im, "Room" + str(i + 1), (locationList[i][0] - 30, locationList[i][1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colorList[i], 2)
        cv2.putText(im, str(round(areaList[i], 2)), (locationList[i][0] - 30, locationList[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i], 2)
        cv2.putText(im, str(round(circuList[i], 2)), (locationList[i][0] - 30, locationList[i][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i], 2)

    cv2.imwrite(output + "contour_" + iml, im)

def main():
    imgList = os.listdir(input)
    if('.DS_Store' in imgList):
        imgList.remove('.DS_Store')
    imgList.sort()
    for iml in imgList:
        print(iml + ":")
        cal_eachArea_eachCir(iml)
        print("Done!!!")

main()