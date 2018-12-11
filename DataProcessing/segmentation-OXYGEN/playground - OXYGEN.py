'''
    Author: JasonL
    Date: 11/21/2018

    Description:
        Preprocessing the OXYGEN floor plan. Generate the plan with only walls.
'''

import cv2
import numpy as np
import random
import os

# kernel used for erosion and dilation
kernel = np.ones((3,3),dtype=np.uint8)
kernel_55 = np.ones((5,5),dtype=np.uint8)

# flip the black and white, may be used in the future
def flip_color(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] > 127:
                img[x][y] = 0
            else:
                img[x][y] = 255

# denoising, remove the blobs that smaller than the threshold area
def denoise(img, area_threshold=10000):
    ret,thresh = cv2.threshold(img,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # fill the small area with white color
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_threshold:
            cv2.drawContours(im2, [c], 0, 0, -1)
    
    return im2

def remove_slash(im):
    kernel1 = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.uint8)
    kernel2 = np.array([[0,0,1],[0,1,0],[1,0,0]],dtype=np.uint8)
    
    ret, im = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
    
    temp1 = cv2.erode(im,kernel1,iterations=10)
    temp1 = cv2.dilate(temp1,kernel,iterations=3)
    temp2 = cv2.erode(im,kernel2,iterations=10)
    temp2 = cv2.dilate(temp2,kernel,iterations=3)
    _, contours1, _ = cv2.findContours(temp1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _, contours2, _ = cv2.findContours(temp2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    area1 = [cv2.contourArea(i) for i in contours1]
    area2 = [cv2.contourArea(i) for i in contours2]
    if np.sum(area1) > np.sum(area2):
        out1 = cv2.erode(im, kernel1, iterations=1)
    else:
        out1 = cv2.erode(im, kernel2, iterations=1)

    out2 = cv2.erode(im,kernel,iterations=1)
    out3 = np.zeros(shape=out2.shape,dtype=np.uint8)
    ind3 = np.logical_and(out1,255-out2)
    out3[ind3] = 255
    out4 = np.zeros(shape=out2.shape,dtype=np.uint8)
    ind4 = np.logical_and(im,255-out3)
    out4[ind4] = 255

    return out4

# preprocess the floor plan -> parameter is for OXYGEN dataset only
def img_preprocessing(img_dir):
    # read the image from the directory
    img_list = os.listdir(img_dir)
    img_list = [x for x in img_list if '.jpg' in x]

    if not os.path.exists('output'):
        os.mkdir('output')
    
    for index in range(len(img_list)):
        
        im = cv2.imread(os.path.join(img_dir, img_list[index]))
        im1 = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

        print(img_list[index])
        
        im_no_slash = remove_slash(im1)

        # first round denoise -> remove the broken lines, etc.
        im_denoise = denoise(im_no_slash, area_threshold=60)
        
        # erosion and dilation
        # 3 steps -> remain the walls with double-line representation
        im_preprocess = cv2.dilate(im_denoise, kernel_55, iterations=4)
        im_preprocess = cv2.erode(im_preprocess, kernel_55, iterations=5)
        im_preprocess = cv2.dilate(im_preprocess, kernel_55, iterations=1)

        # delete the label
        label_pos = [(im1.shape[1] - 600, 0), (im1.shape[1], im1.shape[0])]
        im_no_label = im_preprocess
        cv2.rectangle(im_no_label, label_pos[0], label_pos[1], 0, -1)

        # first round denoise -> remove dots
        im_denoise_2 = denoise(im_no_label, area_threshold=60000)

        # write images into the output directory
        cv2.imwrite('./output/'+str(img_list[index]), im_denoise_2)

img_preprocessing('./Images')