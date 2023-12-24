from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np
import time

show_lines = False

cap = cv2.VideoCapture("Videos/people.mp4") #For video

model = YOLO('../Yolo_weights/yolov8m.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("project_person_counter/mask_up.png")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

up_limit = [150,297,350,297]
down_limit = [450,350,700,350]

up_count = []
down_count = []

# Variables for FPS calculation
start_time = time.time()
frame_counter = 0

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("project_person_counter/graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(700,0))
    
    results = model(imgRegion,stream=True)

    detections = np.empty((0,5))

    # Calculate FPS
    frame_counter += 1
    if (time.time() - start_time) > 1:
        fps = frame_counter / (time.time() - start_time)
        frame_counter = 0
        start_time = time.time()


    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1,y1,x2,y2), (255,0,255), 3)
            
            # x1,y1,w,h = box.xywh[0]
            bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)

            #confidence
            conf = math.ceil(box.conf[0]*100)/100            
            print(conf)

            #Class Name
            cls = int(box.cls[0])
            # print(cls)

            CurrentClass = classNames[cls]

            if (CurrentClass == "person") and conf > 0.3:
                # cvzone.putTextRect(img, f"{CurrentClass} {conf}", ( max(0,int(x1)), max(40,int(y1)) ), 
                #                    scale=0.6, thickness=1, offset = 3)
                # cvzone.cornerRect(img, bbox,l=10,rt=5)
                currentArray = np.array([int(x1),int(y1),int(x2),int(y2),conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    if show_lines:

        cv2.line(img,(up_limit[0], up_limit[1]), (up_limit[2],up_limit[3]),(0,0,255),5)
        cv2.line(img,(down_limit[0], down_limit[1]), (down_limit[2],down_limit[3]),(0,0,255),5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result            
        print(result)
        bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)
        cvzone.cornerRect(img, bbox, l=9, rt = 2, colorR=(255,0,255))
        cvzone.putTextRect(img, f"person: {int(id)}", ( max(0,int(x1)), max(40,int(y1)) ), 
                                   scale=2, thickness=3, offset = 10)
        
        cx, cy = int(x1)+int(x2-x1)//2, int(y1)+int(y2-y1)//2
        # cv2.circle(img, (cx,cy),5,(255,0,255),cv2.FILLED)

        if up_limit[0] < cx < up_limit[2] and up_limit[1] - 15 < cy < up_limit[1] + 15:
            if up_count.count(id) == 0:
                up_count.append(id)
                if show_lines:
                    cv2.line(img,(up_limit[0], up_limit[1]), (up_limit[2],up_limit[3]),(0,255,0),5)

        if down_limit[0] < cx < down_limit[2] and down_limit[1] - 15 < cy < down_limit[1] + 15:
            if down_count.count(id) == 0:
                down_count.append(id)
                if show_lines:
                    cv2.line(img,(down_limit[0], down_limit[1]), (down_limit[2],down_limit[3]),(0,255,0),5)


    # cvzone.putTextRect(img, f"Count: {len(up_count)}", ( 50,50 ))
                
    cv2.putText(img, str(len(up_count)), (900,88), cv2.FONT_HERSHEY_PLAIN, 5, (74,195,139),8)
    cv2.putText(img, str(len(down_count)), (1150,88), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,255),8)

    cv2.putText(img, f"FPS: {int(fps)}", (1000, 680), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)


        

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

