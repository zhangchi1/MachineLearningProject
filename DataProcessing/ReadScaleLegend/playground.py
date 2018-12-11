import cv2
import numpy as np

# 1,142,143

k = np.ones((3,3))

im = cv2.imread("temp3.png")
# cv2.imshow("im",im)


# im1 = im[:,:,0]
# im2 = im[:,:,1]
# im3 = im[:,:,2]
# im1[im1!=1] = 0
# im2[im2!=142] = 0
# im3[im3!=143] = 0

# im[:,:,0] = im1
# im[:,:,1] = im2
# im[:,:,2] = im3

# im = cv2.erode(im,kernel=k)
# im = cv2.dilate(im,kernel=k)
# im = cv2.erode(im,kernel=k)
# im = cv2.dilate(im,kernel=k)

im1 = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

_,thresh = cv2.threshold(im1,200,255,cv2.THRESH_BINARY_INV)

# thresh = cv2.erode(thresh,kernel=k,iterations=10)
# thresh = cv2.dilate(thresh,kernel=k,iterations=10)

_, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(im, contours, -1, (0,0,255), 3)

lines = cv2.HoughLinesP(thresh,1,np.pi/180,100,minLineLength=30,maxLineGap=20)#20
N = lines.shape[0]
for i in range(N):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]
    if abs(y1-y2) <= 5 and abs(x1-x2) <= 500:
        cv2.line(im,(x1,y1),(x2,y2),(255,0,0),1)
    # cv2.line(im,(x1,y1),(x2,y2),(255,0,0),1)


# sum = 0
# for c in contours:
#     sum += cv2.contourArea(c)
# print(sum)
# cv2.imshow("im",im)

_im1 = im[:,:,0]
_im2 = im[:,:,1]
_im3 = im[:,:,2]

l = np.logical_and(_im1==255,np.logical_and(_im2==0,_im3==0))

new_im = np.zeros(shape=im.shape)
new_im[l] = [255,255,255]

cv2.imwrite("output.jpg",new_im)
# cv2.imwrite("im.jpg",im)
# cv2.imshow("im1",im1)
# cv2.imshow("thresh",thresh)

cv2.waitKey(0)