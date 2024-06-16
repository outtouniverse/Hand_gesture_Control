import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

import tensorflow as tf
import ml_dtypes

print(tf.__version__)
print(ml_dtypes.float8_e4m3fn)

# Screen sizes
width, height = 1280, 720  #
w_c, h_c = 720, 480 

folder = "Present"

cap = cv2.VideoCapture(0)
cap.set(3, w_c)
cap.set(4, h_c)

pathImage = sorted(os.listdir(folder), key=len)
print(pathImage)

detector = HandDetector(detectionCon=0.8, maxHands=1)


annotations = []
annotationNumber = -1 
buttonPress = False
buttoncounter = 0
buttonDelay = 20
annotStart = False

gestureThreshold = int(h_c / 2)

i = 0  

while True:
    
    success, img = cap.read()

    
    pathfullimg = os.path.join(folder, pathImage[i])
    imgc = cv2.imread(pathfullimg)

    
    imgc = cv2.resize(imgc, (width, height))

   
    hands, img = detector.findHands(img)

    cv2.line(img, (0, gestureThreshold), (w_c, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPress:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmlist = hand['lmList']

        if lmlist:
            xval = int(np.interp(lmlist[8][0], [0, w_c], [0, width]))
            yval = int(np.interp(lmlist[8][1], [0, h_c], [0, height]))
            indexfinger = xval, yval

            if cy < gestureThreshold:
                annotStart = False

                if fingers == [1, 0, 0, 0, 0]:
                    print("Left")
                    if i > 0:
                        buttonPress = True
                        i -= 1

               
                elif fingers == [0, 0, 0, 0, 1]:
                    print("Right")
                    if i < len(pathImage) - 1:
                        buttonPress = True
                        i += 1
                        annotations = []
                        annotationNumber = -1 
            elif fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgc, indexfinger, 12, (0, 0, 255), cv2.FILLED)
                annotStart = False
            
            elif fingers == [0, 1, 0, 0, 0]:
                if not annotStart:
                    annotStart = True
                    annotations.append([])
                    annotationNumber += 1
                cv2.circle(imgc, indexfinger, 12, (0, 0, 255), cv2.FILLED)
                annotations[annotationNumber].append(indexfinger)
            else:
                annotStart = False

          
            if fingers == [0, 1, 1, 1, 0]:
                if annotations and annotationNumber >= 0:
                    annotations.pop()
                    annotationNumber -= 1
                    buttonPress = True

   
    if buttonPress:
        buttoncounter += 1
        if buttoncounter > buttonDelay:
            buttoncounter = 0
            buttonPress = False

    
    for annotation in annotations:
        for k in range(1, len(annotation)):
            cv2.line(imgc, annotation[k - 1], annotation[k], (0, 0, 200), 12)

    cv2.imshow("Video", img)

    
    cv2.imshow("Slides", imgc)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()
