from embedding import cosine_similarity
import numpy as np
from typing import List, Tuple, Set


class CustomTracker:
    def __init__(self, match_threshold: float = 0.4):
        self.tracks = {}  # {id: embedding}
        self.next_id = 0
        self.match_threshold = match_threshold

    def update(self, detections: list) -> list:
        """
        For each detection (a dict with keys 'box' and 'embedding'),
        check against stored tracks. If a match is found, assign the older (stored) ID.
        Otherwise, assign a new ID.
        """
        for detection in detections:
            best_match_id = None
            best_similarity = -np.inf
            for track_id, stored_embedding in self.tracks.items():
                sim = cosine_similarity(detection["embedding"], stored_embedding)
                if sim > best_similarity and sim >= self.match_threshold:
                    best_similarity = sim
                    best_match_id = track_id
            if best_match_id is not None:
                detection["id"] = best_match_id
                self.tracks[best_match_id] = detection["embedding"]
            else:
                detection["id"] = self.next_id
                self.tracks[self.next_id] = detection["embedding"]
                self.next_id += 1
        return detections


def is_in_checkout_zone(points: np.ndarray, zone: Tuple[int, int, int, int]) -> bool:
    x_min, y_min, x_max, y_max = zone
    box_x_min = min(points[0][0], points[1][0])
    box_y_min = min(points[0][1], points[1][1])
    box_x_max = max(points[0][0], points[1][0])
    box_y_max = max(points[0][1], points[1][1])
    return (
        box_x_min < x_max
        and box_x_max > x_min
        and box_y_min < y_max
        and box_y_max > y_min
    )


def check_line_crossing(
    current_pos: np.ndarray,
    previous_pos: np.ndarray,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> Tuple[bool, str]:
    if previous_pos is None:
        return False, ""
    current_centroid = np.mean(current_pos, axis=0)
    previous_centroid = np.mean(previous_pos, axis=0)

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = current_centroid
    B = previous_centroid
    C = np.array(line_start)
    D = np.array(line_end)
    has_crossed = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    if has_crossed:
        moving_right = current_centroid[0] > previous_centroid[0]
        direction = "entrance" if moving_right else "exit"
        return True, direction
    return False, ""


class ZoneTracker:
    def __init__(self):
        self.currently_in_checkout: Set[int] = set()
        self.has_visited_checkout: Set[int] = set()
        self.alerted_entrance: Set[int] = set()
        self.alerted_exit: Set[int] = set()
        self.frame_count = 0

    def format_timestamp(self, frame_count: int, fps: float) -> str:
        total_seconds = frame_count / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def process_tracked_objects(
        self,
        tracked_objects: list,
        checkout_zone: Tuple[int, int, int, int],
        line_points: Tuple[Tuple[int, int], Tuple[int, int]],
        track_points: str,
        fps: float = 25.0,
    ) -> List[str]:
        alerts = []
        currently_in_checkout = set()
        self.frame_count += 1
        timestamp = self.format_timestamp(self.frame_count, fps)
        for obj in tracked_objects:
            current_points = obj.estimate
            past_points = (
                obj.past_detections[-1].points
                if obj.past_detections
                else current_points
            )

            in_checkout = is_in_checkout_zone(current_points, checkout_zone)
            if in_checkout:
                currently_in_checkout.add(obj.id)
                if obj.id not in self.currently_in_checkout:
                    alerts.append(
                        f"[{timestamp}] ALERT: Person {obj.id} entered checkout zone"
                    )
                    self.has_visited_checkout.add(obj.id)
            if obj.id in self.currently_in_checkout and not in_checkout:
                alerts.append(
                    f"[{timestamp}] ALERT: Person {obj.id} left checkout zone"
                )

            crossed, direction = check_line_crossing(
                current_points, past_points, line_points[0], line_points[1]
            )
            if crossed:
                if direction == "entrance" and obj.id not in self.alerted_entrance:
                    alerts.append(
                        f"[{timestamp}] ALERT: Person {obj.id} entered through entrance line"
                    )
                    self.alerted_entrance.add(obj.id)
                elif direction == "exit" and obj.id not in self.alerted_exit:
                    if obj.id in self.has_visited_checkout:
                        alerts.append(
                            f"[{timestamp}] ALERT: Person {obj.id} exited after visiting checkout"
                        )
                    else:
                        alerts.append(
                            f"[{timestamp}] ALERT: Person {obj.id} exited WITHOUT visiting checkout"
                        )
                    self.alerted_exit.add(obj.id)
        self.currently_in_checkout = currently_in_checkout
        return alerts


class DummyTrackedObject:
    def __init__(
        self,
        obj_id: int,
        current_centroid: Tuple[int, int],
        past_centroid: Tuple[int, int] = None,
    ):
        self.id = obj_id
        self.estimate = np.array(
            [
                [current_centroid[0], current_centroid[1]],
                [current_centroid[0], current_centroid[1]],
            ]
        )
        if past_centroid is not None:
            dummy_det = type("DummyDetection", (), {})()
            dummy_det.points = np.array(
                [
                    [past_centroid[0], past_centroid[1]],
                    [past_centroid[0], past_centroid[1]],
                ]
            )
            self.past_detections = [dummy_det]
        else:
            self.past_detections = []
