from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")      # load a pretrained model (recommended for training)


def mobile_phone_detection(frame):
    # Perform object detection
    results = model(frame, stream=False, verbose=False, classes=[67])

    mobile_phone_locations = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            if confidence < 0.5:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mobile_phone_locations.append([x1, y1, x2, y2])

    return mobile_phone_locations


def batch_mobile_phone_detection(batch_of_frames: list):
    # Perform object detection
    results = model(batch_of_frames, stream=False, verbose=False, classes=[67])

    batch_mobile_phone_locations = []
    for result in results:
        # mobile_phone_locations = []
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            if confidence < 0.5:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # mobile_phone_locations.append([x1, y1, x2, y2])
            batch_mobile_phone_locations.append([x1, y1, x2, y2])

    return batch_mobile_phone_locations
