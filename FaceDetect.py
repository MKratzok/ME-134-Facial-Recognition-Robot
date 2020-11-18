import numpy as np
import cv2
import time
from robot import Robot


def color_detect(c_img):
    # Convert the img in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(c_img, cv2.COLOR_BGR2HSV)

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
    res_blue = cv2.bitwise_and(c_img, c_img,
                               mask=blue_mask)

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            print("Sees blue")
            return True
        else:
            print("Not blue")
            return False

    return False

rob = Robot()

# Good Theory
# https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1

# Other cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer size to prevent lag
cap.set(cv2.CAP_PROP_FPS, 10) # lower FPS to minimize lag
time.sleep(0.25)

prev = time.time()

while 1:
    ret, img = cap.read()

    if time.time() - prev >= 0.1:
        prev = time.time()
        print('-------------------------')
        print('Commencing Scan')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # We want to pull out the first value in faces, but I'm not sure how to do it
        # print(faces[0])

        if len(faces) == 0:
            print('-------------------------')
            print("No Face Detected")
        else:
            print('-------------------------')
            print("Face Detected")
            # print(faces[0])

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            if not color_detect(img[y:y+h, x:x+w]):
                rob.crawl()

            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                print('-------------------------')
                print("No Eyes Detected")
            else:
                print('-------------------------')
                print("Eyes Detected")

                if len(eyes) == 1:
                    print('Are you winking?')
                    rob.dance(5)
                else:
                    if eyes[0][1] - eyes[0][3]/2 > 10 + eyes[1][1] - eyes[1][3]/2:
                        print('Tilt Left')
                        rob.turn(45)
                    elif eyes[1][1] - eyes[1][3]/2 > 10 + eyes[0][1] - eyes[0][3]/2:
                        print('Tilt Right')
                        rob.turn(-45)
                    else:
                        print('No Tilt')

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # cv2.imshow('img',img)   # This shows the video feed in real time, comment out when running headless
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Break when 'q' is pressed
       break

cap.release()
cv2.destroyAllWindows()


