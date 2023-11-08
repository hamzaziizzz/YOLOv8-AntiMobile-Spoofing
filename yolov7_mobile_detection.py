import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import numpy as np  # Import NumPy

# Load YOLOv7 model
weights = 'yolov8n.pt'  # Replace with the path to your YOLOv7 weights file
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load(weights, map_location=device)
model.eval()


def mobile_phone_detection(frame):
    # Preprocess the frame
    img = frame[:, :, ::-1].transpose(2, 0, 1)
    # print(img.shape)

    # Create a contiguous copy of the array
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        detections = model(img)[0]

    # Apply non-maximum suppression to remove redundant detections
    results = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.4, classes=None, agnostic=False)

    return results


def batch_mobile_phone_detection(batch_of_frames: list):
    # Preprocess the frames
    batch = []

    for frame in batch_of_frames:
        img = frame[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().div(255.0)
        batch.append(img)

    # Stack the frames into a batch
    img_batch = torch.stack(batch, dim=0)

    # Perform object detection on the batch
    with torch.no_grad():
        detections = model(img_batch)

    results_batch = []
    for detections_per_frame in detections:
        # Apply non-maximum suppression to remove redundant detections
        results = non_max_suppression(detections_per_frame, conf_thres=0.5, iou_thres=0.4, classes=[67], agnostic=False)
        results_batch.append(results)

    return results_batch
