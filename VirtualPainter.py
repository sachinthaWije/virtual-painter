import cv2
import os
import numpy as np
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 100

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
selectedColor = (100, 100, 100)

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
print(cap.get(3))
print(cap.get(4))

detector = htm.HandDetector(min_detection_confidence=0.95)
xp, yp = 0, 0
imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        print(x1)
        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    selectedColor = (250, 0, 250)
                if 550 < x1 < 750:
                    header = overlayList[1]
                    selectedColor = (250, 100, 0)
                if 850 < x1 < 900:
                    header = overlayList[2]
                    selectedColor = (5, 255, 100)
                if 1050 < x1 < 1200:
                    header = overlayList[3]
                    selectedColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 - 15), selectedColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            print("Drawing Mode")
            cv2.circle(img, (x1, y1), 15, selectedColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if selectedColor ==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), selectedColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), selectedColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), selectedColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), selectedColor, brushThickness)
            xp, yp = x1, y1


    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)

    img[0:125, 0:1280] = header
    # img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
