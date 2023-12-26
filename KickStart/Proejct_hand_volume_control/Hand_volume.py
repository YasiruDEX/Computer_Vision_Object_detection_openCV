import cv2
import HandTrackingModule as htm
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


cap = cv2.VideoCapture(0) #For webcam
cap.set(3,640)
cap.set(4,480)

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.Getmute()
# volume.GetMasterVolumeLevel()
volRange = (volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(-20.0, None)
minVol = volRange[0]
maxVol = volRange[1]

volBar = 400
volPercent = 0

Color = (255, 0, 255)

def blend(list_images): # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output

def Change_color_with_value(value, minimum, maximum):
    # Normalize the value between 0 and 1
    normalized_value = (value - minimum) / (maximum - minimum)
    
    # Calculate hue value (from 0 to 360 for full spectrum)
    hue = normalized_value * 360
    
    # Convert HSV to RGB
    hue /= 60
    i = int(hue)
    f = hue - i
    p = 1 - 1
    q = 1 - f
    t = 1 if i % 2 == 0 else f
    r, g, b = 0, 0, 0
    
    if i == 0:
        r, g, b = 1, t, p
    elif i == 1:
        r, g, b = q, 1, p
    elif i == 2:
        r, g, b = p, 1, t
    elif i == 3:
        r, g, b = p, q, 1
    elif i == 4:
        r, g, b = t, p, 1
    else:
        r, g, b = 1, p, q
    
    # Scale RGB values to 255 (8-bit color)
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    
    return (r, g, b)


while True:
    success, img1 = cap.read()
    img = detector.findHands(img1)

    lmList = detector.findPosition(img, draw=False)
    if len(lmList[0]) != 0:
        # print(lmList[0][4], lmList[0][8])

        cv2.circle(img, (lmList[0][4][1], lmList[0][4][2]), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (lmList[0][8][1], lmList[0][8][2]), 10, (255, 0, 255), cv2.FILLED)

        cx = int((lmList[0][4][1]+lmList[0][8][1])//2)
        cy = int((lmList[0][4][2]+lmList[0][8][2])//2)

        cv2.line(img, (lmList[0][4][1], lmList[0][4][2]), (lmList[0][8][1], lmList[0][8][2]), Color, 3)
                
        Color = Change_color_with_value(volPercent,0,100)
        print(Color)

        cv2.circle(img, (cx, cy), 10, Color, cv2.FILLED)
    
        length = math.hypot(lmList[0][4][1]-lmList[0][8][1], lmList[0][4][2]-lmList[0][8][2])
        # print(length)

        # if length < 50:
        #     cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        # Hand range 15 - 255
        # Volume range -65 - 0
        vol = np.interp(length, [15, 255], [minVol, maxVol])
        volBar = np.interp(length, [15, 255], [400, 150])
        volPercent = np.interp(length, [15, 255], [0, 100])

        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

    #volumebar in purple inside green outline
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), Color, cv2.FILLED)

    #volume percentage
        #stroke
    cv2.putText(img, f'{int(volPercent)} %', (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, f'{int(volPercent)} %', (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (volPercent*2, 0, volPercent*2), 2)
    

    cv2.imshow("Image", img)
    cv2.waitKey(1)

