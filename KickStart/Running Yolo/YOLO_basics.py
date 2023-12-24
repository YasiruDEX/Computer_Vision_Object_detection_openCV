from ultralytics import YOLO
import cv2

model = YOLO('../Yolo_weights/yolov8n.pt')

results = model("Running YOLO/Images/3.png",show=True)
cv2.waitKey(0)

