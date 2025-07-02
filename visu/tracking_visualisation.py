import cv2
import numpy as np
from processing.inference import get_class_names

def draw_track_history(frame, tracks, detections, history_length=300):
    class_names = get_class_names()
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        color = get_color(int(track_id))
        # Maintain a global dictionary to store track histories
        if not hasattr(draw_track_history, "track_histories"):
            draw_track_history.track_histories = {}
        track_histories = draw_track_history.track_histories

        # Use the center of the bounding box for the history point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if track_id not in track_histories:
            track_histories[track_id] = []
        track_histories[track_id].append((cx, cy))
        # Keep only the last 'history_length' points
        history = track_histories[track_id][-history_length:]
        for i in range(len(history) - 1):
            pt1 = (int(history[i][0]), int(history[i][1]))
            pt2 = (int(history[i + 1][0]), int(history[i + 1][1]))
            cv2.line(frame, pt1, pt2, color, 2)

        # Draw a colored bounding box for the latest entry in the object's track
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Find the best matching detection for this track (IoU or center proximity)
        best_det = None
        best_dist = float('inf')
        for det in detections:
            (dx, dy, dw, dh), conf, cls = det
            dcx = dx + dw // 2
            dcy = dy + dh // 2
            dist = (dcx - cx) ** 2 + (dcy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_det = det
        if best_det is not None:
            (_, _, _, _), conf, cls = best_det
            class_name = class_names.get(cls, str(cls))
            label = f'ID {track_id} {class_name} {conf:.2f}'
        else:
            label = f'ID {track_id}'
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    return frame

def get_color(idx):
    np.random.seed(idx)
    color = tuple(int(x) for x in np.random.randint(0, 255, 3))
    return color

def reset_track_histories():
    if hasattr(draw_track_history, "track_histories"):
        draw_track_history.track_histories = {}