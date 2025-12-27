import cv2
import numpy as np
import sys
import os

# Add local libs to path
libs_path = os.path.abspath('libs')
if libs_path not in sys.path:
    sys.path.append(libs_path)

from ultralytics import YOLO

# Initialize YOLOv8 model
# It will download yolov8n.pt on first run
model = YOLO('yolov8n.pt')

# Video Capture
cap = cv2.VideoCapture('video.mp4')

count_line_position = 550
offset = 6 # Range for line crossing
counter = 0
counter_set = set() # To store IDs of vehicles already counted

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking
    # persist=True is essential for tracking objects across frames
    results = model.track(frame, persist=True, verbose=False)
    
    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (1270, count_line_position), (255, 127, 0), 3)

    # Process detections
    # results[0].boxes contains the detection boxes
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu() # x, y, width, height (center x, center y)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            # COCO Class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
            if class_id in [2, 3, 5, 7]:
                x, y, w, h = box
                cx = int(x)
                cy = int(y)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Draw bounding box logic (optional, YOLO can plot, but we draw custom)
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Counting Logic
                if count_line_position - offset < cy < count_line_position + offset:
                    if track_id not in counter_set:
                        counter += 1
                        counter_set.add(track_id)
                        cv2.line(frame, (25, count_line_position), (1270, count_line_position), (0, 127, 255), 3)
                        print(f"Vehicle Count: {counter}")

    # Display Count
    cv2.putText(frame, "Vehicle Counter : " + str(counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("YOLOv8 Vehicle Counter", frame)
    
    # WaitKey 1 for normal speed (YOLO inference takes time so 1ms is fine)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()