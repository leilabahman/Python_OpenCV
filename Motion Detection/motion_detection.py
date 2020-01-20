# -*- coding: utf-8 -*-
"""
Basic Motion Detection

Created on Mon Jan 20 09:45:33 2020

@author: leiba
"""

import cv2
import datetime
import time
import imutils


def motion_detection():
    #Define 0 for selecting default camera 
    video_capture=cv2.VideoCapture(0)
    time.sleep(2)
    
    #Initiate the first frame
    first_frame=None
    
    #Loop through frames
    while True:
        # Video_capture.read gives two output (retval & frame). 
        # In order to select the threshold image, we put [1]
        frame=video_capture.read()[1]
        text="No motion detected"
        
        # Covert the frame to grayscale
        grayscale_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gaussian_frame=cv2.GaussianBlur(grayscale_frame,(21,21),0)
        blur_image=cv2.blur(gaussian_frame,(5,5))
        
        if first_frame is None:
            first_frame=blur_image
        else:
            pass
        
        frame=imutils.resize(frame,width=500)
        
        # Compute the absolute difference between each pixel between 
        # the first frame and blur image
        frame_delta=cv2.absdiff(first_frame,blur_image)
        
        #threshold gives two output (retval & threshold image). 
        #In order to select the threshold image, we put [1]
        thresh=cv2.threshold(frame_delta,10,255,cv2.THRESH_BINARY)[1]
        
        #Expand white pixels in binary image twice with default kernel 3x3
        dilate_image=cv2.dilate(thresh,None,iterations=3)
        
        #Find contours in binary image
        # Gives contours & hierachy.So we use [1] to get just the countours
        # cv2.CHAIN_APPROX_SIMPLE saves memory by removing all redundant points
        contours=cv2.findContours(dilate_image.copy(),cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[1]

        
        #Loop over the contours
        for c in contours:
            #Draw bounding box for countors with area greater than 800
            # You might change the threshold base on the room light
            if cv2.contourArea(c)>800:
                x,y,w,h=cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                text="Motion Detected"            
            else:
                pass
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,f"[+] {text}",(10,20),font,0.5,(0,0,255),2)
        
        # Show current time on the frame        
        cv2.putText(frame,datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M:%S")
        ,(10,frame.shape[0]-10) ,font,0.5,(0,0,255),2)
        
        
        #Demonstrate the frames
        cv2.imshow("Scene Status",frame)
        cv2.imshow("Foreground Mask", dilate_image)
        cv2.imshow("Frame_delta", frame_delta)
        
        key=cv2.waitKey(1)& 0xFF
        if key==ord("q"):
            cv2.destroyAllWindows()
            break
        
if __name__=="__main__":
    motion_detection()
            
        
        
        
        
        
        #
        
        
            
    
    
