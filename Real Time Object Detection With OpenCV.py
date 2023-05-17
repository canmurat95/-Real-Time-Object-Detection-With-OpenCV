
from cv2 import putText
from imutils import contours

import regex
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import datetime
import sys
import time
from datetime import timedelta
import re
from decimal import Decimal
from sympy import per
import math
import locale








cap = cv2.VideoCapture('traffic.mp4')
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.
fps = 0
if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))



sec = 0
period = '00'
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

backSub = cv2.createBackgroundSubtractorMOG2()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
firstFrame=0;
currentRoi=[];
firstFrames=[];


kernelErode = np.ones((1,1), np.uint8)
kernelDilate = np.ones((5,5), np.uint8)

kernelErodeMotion = np.ones((3,3), np.uint8)
kernelDilateMotion = np.ones((5,5), np.uint8)

while(cap.isOpened()): 
        
    # Capture frame-by-frame 
    ret, frame = cap.read() 



    
    if ret == True: 

        cfn = cap.get(1)
        if int(cfn)%int(fps)==0:
        

           
            period = "{:2d}".format(sec)
            sec = sec + 1
            #
            td=float(period)
            td1=td/fps
            period1=format(td1)
            #
            td2 =float(period)
            td3=td2/fps
            period2=format(td3)
          
            font = cv2.FONT_HERSHEY_SIMPLEX
       
            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
    # Set range for red color and 
    # define mask
            red_lower = np.array([136, 87, 111], np.uint8)
            red_upper = np.array([180, 255, 255], np.uint8)
            red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
  
 
    # Set range for blue color and
    # define mask

      
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
            kernal = np.ones((5, 5), "uint8")
            
    # For red color
            red_mask = cv2.dilate(red_mask, kernal)
            res_red = cv2.bitwise_and(frame, frame, 
                                    mask = red_mask)
            
    
  
   
    # Creating contour to track red color
            contours, hierarchy = cv2.findContours(red_mask,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
            
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area > 300):
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(frame, (x, y), 
                                            (x + w, y + h), 
                                            (0, 0, 255), 2)
                    cv2.putText(imageFrame, "Stop", (125, 50 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
                    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)

    frameChange = cv2.GaussianBlur(frame, (3, 3),0)



    # 1- convert frame from BGR to HSV
    setNewROI=0;
    if cv2.waitKey(1) & 0xFF == ord('s'):
        currentRoi.append(cv2.selectROI("select",frame));
        cv2.destroyWindow('select')
        setNewROI=1; 
            
    for i in range(len(currentRoi)):
    
        
        rect=currentRoi[i];
        start_point = (int(rect[0]), int(rect[1]))
        end_point = (int(rect[0])+int(rect[2]), int(rect[1])+int(rect[3]))

        mask = np.zeros(frame.shape, np.uint8)     
    
        cv2.rectangle(mask, start_point, end_point, (255, 255, 255), -1)
        result = cv2.bitwise_and(frame, mask)
        
        
        img_yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        if(setNewROI==1 and (len(currentRoi)-1)==i):
            firstFrames.append(result);
            continue;

        #firstImage = cv2.cvtColor(firstFrames[i], cv2.COLOR_BGR2GRAY)
        #currentImage = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        #fgMask = cv2.subtract(firstImage, currentImage)

        
       
        #cv2.imshow("firstImage", firstImage)
        #cv2.imshow("currentImage", currentImage)
      
        fgMask = backSub.apply(result,100);
        ret,fgMask = cv2.threshold(fgMask,25,255,cv2.THRESH_BINARY)

        fgMaskErode = cv2.erode(fgMask, kernelErodeMotion, iterations=1)
        fgMask = cv2.dilate(fgMaskErode,kernelDilateMotion, iterations=1)
        

        contours,hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            for pic,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if(area > 10): #tSo remove the noise
                    
                    
                    
                     
                    print("Center HIT:"+str(len(contours))+"Current ROI:"+str(len(currentRoi)))
                   
                    cv2.rectangle(frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (0, 0, 255), 4)
                   
                    cv2.putText(frame,period1,(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)  
                    print("Toplam Çalışma süresi:"+str(period1))
                    
                    

                    break;
           
            '''
            print("Center HIT:"+str(len(contours))+"Current ROI:"+str(len(currentRoi)))
            cv2.rectangle(frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (0, 0, 255), 4)                
            '''
        #else:
            #firstFrames[i]=result          
            '''
            for pic,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area > 20): #to remove the noise
                    print("Center HIT:"+str(len(contours))+"Current ROI:"+str(len(currentRoi)))
                    cv2.rectangle(frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (0, 0, 255), 4)
                    isHitNewFrame=1;
                    break;
            '''        
       
         
                  
            
        
        cv2.imshow("fgMask", fgMask)
        cv2.imshow("result", result)
        centerPoint=(int((int(start_point[0])+int(end_point[0]))/2),int((int(start_point[1])+int(end_point[1]))/2))
      
        cv2.rectangle(frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)
        
       
        cv2.putText(frame,str(i+1)+".Makina", (int(start_point[0]), int(start_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150))
        
        # Display the resulting frame 
      
        
     
        
   
   
    HSV = cv2.cvtColor(frameChange,cv2.COLOR_BGR2HSV)

    #low_red=np.array([0, 100, 250])
    #high_red=np.array([1, 255, 255])

    #low_orange=np.array([25, 50, 250])
    #high_orange=np.array([30, 255, 255])

    low_green = np.array([70, 52, 150])
    high_green = np.array([90, 255, 255])
    
    #check if the HSV of the frame is lower or upper red
   
    
    Green_mask = cv2.inRange(HSV,low_green, high_green)
    #Red_mask = cv2.inRange(HSV,low_red, high_red)
    #Orange_mask = cv2.inRange(HSV,low_orange, high_orange)

    #Green_mask = cv2.erode(Green_mask, kernelErode, iterations=1)
    Green_mask = cv2.dilate(Green_mask,kernelDilate, iterations=1)
    #Red_mask = cv2.dilate(Reds_mask,kernelDilate, iterations=1)
    #Orange_mask = cv2.dilate(Orange_mask,kernelDilate, iterations=1)

    
    # Draw rectangular bounded line on the detected red area
    contours,hierarchy = cv2.findContours(Green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #if((len(contours))>0):
      #print("Green Contour:"+ str(len(contours)));

    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 0): #to remove the noise
          
            # Constructing the size of boxes to be drawn around the detected red area
            x,y,w,h = cv2.boundingRect(contour)
            

       
         
            #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)
            cv2.circle(frame,(x, y), 4, (0,255,0), 4)
           
    
    '''
    contours,hierarchy = cv2.findContours(Red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print("Red Contour:"+ str(len(contours)));
    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 0): #to remove the noise
            # Constructing the size of boxes to be drawn around the detected red area
            x,y,w,h = cv2.boundingRect(contour)
            #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)
            cv2.circle(frame,(x, y), 2, (0,0,255), 2)
    
    
    
    contours,hierarchy = cv2.findContours(Orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #print("Orange Contour:"+ str(len(contours)));
    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 0): #to remove the noise
            # Constructing the size of boxes to be drawn around the detected red area
            x,y,w,h = cv2.boundingRect(contour)
            #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)
            cv2.circle(frame,(x, y), 4, (0,255,255),2)          
    '''


    cv2.imshow("Tracking Green Color",frame)
    
    #cv2.imshow("Mask_RED",Red_mask)
    cv2.imshow("Mask_Green",Green_mask)
    #cv2.imshow("Mask_Orange",Orange_mask)
    #cv2.imshow('FG Mask', fgMask)
    #cv2.imshow("And",result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()