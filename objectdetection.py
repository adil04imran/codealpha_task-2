import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set up video capture (0 is for webcam)
cap = cv2.VideoCapture(0)

# Initialize CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Variable to keep track of whether we are tracking
tracking = False
bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection using YOLOv5
    results = model(frame)
    detections = results.xywh[0].cpu().numpy()  # [x_center, y_center, width, height, confidence, class]
    
    # Draw detections from YOLOv5
    for det in detections:
        x_center, y_center, width, height, conf, _class = det
        if conf > 0.5:  # Confidence threshold
            # Convert YOLOv5 bbox format (center_x, center_y, width, height) to (x, y, width, height)
            x = x_center - width / 2
            y = y_center - height / 2
            cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # If no tracker is running, start tracking the first detected object
            if not tracking:
                bbox = (int(x), int(y), int(width), int(height))
                tracker.init(frame, bbox)
                tracking = True

    # If tracker is active, update the tracking
    if tracking and bbox:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, f"Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            tracking = False  # Stop tracking if object is lost

    # Display the frame
    cv2.imshow('Object Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
