from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# cap = cv2.VideoCapture(0) #For webcam
# cap.set(3,1280)
# cap.set(4,720)

cap = cv2.VideoCapture("Videos/ppe-3.mp4") #For video

model = YOLO('Project_PPE_detection/ppe_trained.pt')

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Variables for FPS calculation
start_time = time.time()
frame_counter = 0
fps = 0

Warnings = []

while True:
    success, img = cap.read()
    results = model(img,stream=True)

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

            Class_name = classNames[cls]

            if conf > 0.5:
                if("NO-" in Class_name):
                    cvzone.cornerRect(img, bbox, l=9, rt = 2, colorR=(0,0,255))
                    cvzone.putTextRect(img, f"{classNames[cls]}", ( max(0,int(x1)), max(40,int(y1)) ),
                                        scale=1, thickness=1,colorR=(0,0,255))
                    Warnings.append(Class_name)

                elif Class_name == "Person":
                    cvzone.cornerRect(img, bbox, l=9, rt = 2, colorR=(255,0,255))
                    cvzone.putTextRect(img, f"{classNames[cls]}", ( max(0,int(x1)), max(40,int(y1)) ),
                                        scale=1, thickness=1) 
                else:
                    cvzone.cornerRect(img, bbox, l=9, rt = 2, colorR=(0,255,0))
                    cvzone.putTextRect(img, f"{classNames[cls]}", ( max(0,int(x1)), max(40,int(y1)) ),
                                        scale=1, thickness=1,colorR=(0,255,0)) 
                

    Warnings = list(dict.fromkeys(Warnings))
    if len(Warnings) > 0:
        cvzone.putTextRect(img, "Warning", (10, 80), scale=4, thickness=3 ,colorR=(0,0,255))
    else:
        cvzone.putTextRect(img, "Safe", (25, 80), scale=2, thickness=2 ,colorR=(0,255,0))
    # cv2.putText(img, f"{Class_name}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
    Warnings = []

    cv2.putText(img, f"FPS: {int(fps)}", (1000, 680), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)


