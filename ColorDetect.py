# Python code for Multiple Color Detection 
  
import numpy as np 
import cv2 
  
  
# Capturing video through webcam 
cap = cv2.VideoCapture(0)
  
# Start a while loop 
while(1): 
      
    # Reading the video from the 
    # webcam in image frames 
    ret, img = cap.read()
  
    # Convert the img in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
  
    # Set range for blue color and 
    # define mask 
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
      
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between img and mask determines 
    # to detect only that particular color 
    kernal = np.ones((5, 5), "uint8") 
      
      
    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernal) 
    res_blue = cv2.bitwise_and(img, img, 
                               mask = blue_mask) 
   
  
    # Creating contour to track blue color 
    contours, hierarchy = cv2.findContours(blue_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            print("Sees blue")
            x, y, w, h = cv2.boundingRect(contour) 
            img = cv2.rectangle(img, (x, y), 
                                       (x + w, y + h), 
                                       (255, 0, 0), 2) 
              
            cv2.putText(img, "Blue Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 0, 0)) 
        else:
            print("Not blue")
        
              
    # Program Termination 
    #cv2.imshow("Multiple Color Detection in Real-TIme", img) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows() 
        break