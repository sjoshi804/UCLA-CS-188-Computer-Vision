import numpy as np
import cv2

#Initializing variables for video
cap = cv2.VideoCapture('video.mp4')
count = 0

#Loop through video
while (cap.isOpened()):
    #Read in frame
    ret, frame = cap.read()
    if not ret:
        break
    
    #Print frame
    name = 'frames/frame' + str(count) + '.jpg'
    print("Writing " + name)
    cv2.imwrite(name, frame)
    count += 1

    #Make image grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()
