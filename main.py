from tracking import CustomTracker, DummyTrackedObject, ZoneTracker
from detection import YOLO, yolo_detections_to_norfair_detections
from embedding import get_embedding
import torch
import numpy as np
import cv2


def run_tracking_pipeline(
    input_video: str, output_video: str, match_threshold: int = 0.35
):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file:", input_video)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    yolo_model = YOLO(
        "yolov7.pt", device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    track_points = "bbox"
    model_threshold = 0.4
    classes = [0]

    tracker = CustomTracker(match_threshold=match_threshold)
    trajectories = {}

    prev_centroids = {}

    zone_tracker = ZoneTracker()
    checkout_zone = (874, 300, 1120, 1080)
    entry_exit_line = ((850, 80), (700, 204))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yolo_dets = yolo_model(
            frame,
            conf_threshold=model_threshold,
            iou_threshold=0.6,
            image_size=720,
            classes=classes,
        )

        norfair_dets = yolo_detections_to_norfair_detections(
            yolo_dets, track_points=track_points
        )

        tracker_detections = []
        for det in norfair_dets:
            x1 = int(det.points[0][0])
            y1 = int(det.points[0][1])
            x2 = int(det.points[1][0])
            y2 = int(det.points[1][1])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            cutout = frame[y1:y2, x1:x2]
            embedding = get_embedding(cutout)
            tracker_detections.append({"box": [x1, y1, x2, y2], "embedding": embedding})

        tracked_detections = tracker.update(tracker_detections)

        trajectories = {}
        last_active_frame = {}
        frame_count = 0

        active_ids = set()
        for det in tracked_detections:
            obj_id = det["id"]
            active_ids.add(obj_id)
            last_active_frame[obj_id] = frame_count

        max_inactive_frames = 30
        to_remove = []
        for obj_id in list(trajectories.keys()):
            if obj_id not in active_ids:
                if frame_count - last_active_frame.get(obj_id, 0) > max_inactive_frames:
                    to_remove.append(obj_id)
        for obj_id in to_remove:
            del trajectories[obj_id]
            if obj_id in last_active_frame:
                del last_active_frame[obj_id]

        for det in tracked_detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            obj_id = det["id"]
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append((cx, cy))
            if len(trajectories[obj_id]) > 5:
                trajectories[obj_id] = trajectories[obj_id][-5:]

        dummy_tracked_objects = []
        for det in tracked_detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            obj_id = det["id"]
            past_centroid = prev_centroids.get(obj_id, None)
            dummy_obj = DummyTrackedObject(obj_id, (cx, cy), past_centroid)
            dummy_tracked_objects.append(dummy_obj)
            prev_centroids[obj_id] = (cx, cy)

        alerts = zone_tracker.process_tracked_objects(
            dummy_tracked_objects, checkout_zone, entry_exit_line, track_points, fps
        )

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        x_min, y_min, x_max, y_max = checkout_zone
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)

        cv2.line(canvas, entry_exit_line[0], entry_exit_line[1], (0, 255, 255), 2)

        for obj_id, pts in trajectories.items():
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (255, 255, 255), 2)

            if pts:
                cv2.circle(canvas, pts[-1], 5, (0, 255, 0), -1)
                cv2.putText(
                    canvas,
                    f"ID: {obj_id}",
                    (pts[-1][0] + 10, pts[-1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        y0 = 30
        for alert in alerts:
            cv2.putText(
                canvas, alert, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            y0 += 25
            print(alert)

        out.write(canvas)

    cap.release()
    out.release()
    print("Tracking complete. Output saved to", output_video)


if __name__ == "__main__":
    run_tracking_pipeline(
        "client-vid.mp4", "visualization-tracks-zones.mp4", match_threshold=0.35
    )
