import cv2
import numpy as np
# 1,142,143

k = np.ones((3,3))

im = cv2.imread("im.jpg")

im1 = im[:,:,0]
im2 = im[:,:,1]
im3 = im[:,:,2]
im1[im1!=1] = 0
im2[im2!=142] = 0
im3[im3!=143] = 0

im[:,:,0] = im1
im[:,:,1] = im2
im[:,:,2] = im3

im = cv2.erode(im,kernel=k)
im = cv2.dilate(im,kernel=k)
im = cv2.erode(im,kernel=k)
im = cv2.dilate(im,kernel=k)

im1 = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

_,thresh = cv2.threshold(im1,50,255,cv2.THRESH_BINARY)

im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im, contours, -1, (0,0,255), 3)

sum = 0
for c in contours:
    sum += cv2.contourArea(c)

print(sum)
cv2.imshow("frame",im)
cv2.imshow("frame1",thresh)

cv2.waitKey(0)