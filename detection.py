import os
import torch
import norfair
import numpy as np

class YOLO:
    def __init__(self, model_path: str, device: str = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(model_path):
            os.system(
                f"wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{os.path.basename(model_path)} -O {model_path}"
            )
        try:
            self.model = torch.hub.load("WongKinYiu/yolov7", "custom", model_path)
        except Exception as e:
            raise Exception("Failed to load model from {}: {}".format(model_path, e))
        self.device = device

    def __call__(self, img, conf_threshold: float = 0.25, iou_threshold: float = 0.45, image_size: int = 720, classes: list = None):
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections

def yolo_detections_to_norfair_detections(yolo_detections, track_points: str = "centroid"):
    """Convert YOLO detections to norfair detections."""
    norfair_detections = []
    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array([[detection_as_xywh[0].item(), detection_as_xywh[1].item()],
                                 [detection_as_xywh[0].item(), detection_as_xywh[1].item()]])
            scores = np.array([detection_as_xywh[4].item(), detection_as_xywh[4].item()])
            norfair_detections.append(norfair.Detection(points=centroid, scores=scores))
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array([[detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                             [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]])
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(norfair.Detection(points=bbox, scores=scores))
    return norfair_detections