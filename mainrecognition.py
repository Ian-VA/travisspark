import cv2 # the opencv library
import imutils
import os
import re
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

capture = cv2.VideoCapture('video.mp4')
frameNr = 0

while True:
    success, frame = capture.read()

    if success:
        cv2.imwrite('frames/{frameNr}.png', frame)
    else:
        break

    frameNr = frameNr + 1

capture.release()

col_frames = os.listdir('frames/') # frames folder, real-time footage split up
col_images=[]

for i in col_frames:
    img = cv2.imread('frames/'+i)
    col_images.append(img) # read images with opencv and append them to col_images

kernel = np.ones((4,4),np.uint8) # 4x4 matrix for image dilation

font = cv2.FONT_HERSHEY_SIMPLEX

pathIn = "contour_frames/"

for i in range(len(col_images)-1): # loop over all images in the video
    
    grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY) # convert the current frame and the next one to grayscale
    grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA) # get the difference between two frames above to look for motion
    
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY) #image thresholding: gets rid of everything that has not changed between frames i and frames i + 1
    
    dilated = cv2.dilate(thresh,kernel,iterations = 1) # image dilation: allows small fragments of moving objects to be more clumped up together by multiplying with a 4x4 matrix

    
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # find the contours, or borders, of a moving object
    
    valid_cntrs = [] # list of valid contours
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr) # get the dimensions of the contour on an xy plane of the image (width and height are also properties)
        if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):  # if the contour is large enough and below our line, it is valid
            if (y >= 90) & (cv2.contourArea(cntr) < 40):
                break
            valid_cntrs.append(cntr)
            
    dmy = col_images[i].copy() # just copy some images for drawing on
    cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)  # draw the actual contours on the image
    
    cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2) #visualization

    if len(valid_cntrs) >= 1:
        print("Intrusion detected")


    cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
    cv2.imwrite(pathIn+str(i)+'.png',dmy)

pathOut = 'vehicle_detection.mp4'
fps = 14.0
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

for i in range(len(files)):
    filename=pathIn + files[i]
    
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])

out.release()
