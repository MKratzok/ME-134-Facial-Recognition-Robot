import numpy as np
import cv2
import time

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size to prevent lag
cap.set(cv2.CAP_PROP_FPS, 1)

prev = time.time()

while 1:
    time_elapsed = time.time() - prev

    ret, img = cap.read()

    if time_elapsed > 1:
        prev = time.time()

        cv2.imshow('img', img)

        
cap.release()
cv2.destroyAllWindows()