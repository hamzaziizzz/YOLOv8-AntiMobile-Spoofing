from ultralytics import YOLO
import cv2
import math

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)
classNames = ["cell phone"]

# Perform object detection
cap = cv2.VideoCapture("rtsp://grilsquad:grilsquad@192.168.18.93:554/stream1")

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    results = model(img, stream=True, verbose=False, classes=[67])
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # w, h = x2 - x1, y2 - y1
            # # Confidence
            # conf = math.ceil((box.conf[0] * 100)) / 100
            # # Class Name
            # cls = int(box.cls[0])
            # if conf > 0.9:
            #
            #     if classNames[cls] == 'cell phone':
            #         color = (0, 255, 0)
            #     else:
            #         color = (0, 0, 255)

    # Display the frame
    cv2.imshow('frame', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
