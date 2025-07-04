import cv2
from processing.inference import get_class_names
import numpy as np

def draw_yolo_detections(frame, detections, fade_frames=12):
    # Optionally avoid modifying the original frame
    frame = frame.copy()
    class_names = get_class_names()

    # Early exit if no detections
    if not detections:
        return frame

    # Update bbox history: add new, age old, remove too old
    history = draw_yolo_detections.bbox_history
    for entry in history:
        entry[-1] += 1  # age += 1

    # Add new detections to history
    for det in detections:
        (x, y, w, h), conf, cls = det
        history.append([x, y, w, h, conf, cls, 0])

    # Remove old entries
    history[:] = [b for b in history if b[-1] < fade_frames]

    # Define class-dependent colors (BGR)
    player_colour = (200, 217, 37)      # player (RGB 37,217,200 -> BGR 200,217,37)
    disc_colour = (114, 38, 249)        # disc (RGB 249,38,114 -> BGR 114,38,249)

    # Batch process detections by age to reduce OpenCV calls
    recent_detections = []
    faded_detections = []
    
    for x, y, w, h, conf, cls, age in history:
        x2, y2 = x + w, y + h
        alpha = max(0.0, 1.0 - (age / fade_frames))

        # Determine color by class name
        class_name = class_names.get(cls, str(cls)).lower()
        if "disc" in class_name:
            color = disc_colour
        else:
            color = player_colour

        detection_data = (x, y, x2, y2, w, h, conf, cls, age, alpha, color)
        
        if age == 0:
            recent_detections.append(detection_data)
        else:
            faded_detections.append(detection_data)

    # Draw faded detections first
    for x, y, x2, y2, w, h, conf, cls, age, alpha, color in faded_detections:
        # Only blend the ROI for the bbox
        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            continue
        overlay = roi.copy()
        thickness = 2
        cv2.rectangle(overlay, (0, 0), (w, h), color, thickness)
        blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
        frame[y:y2, x:x2] = blended

    # Draw recent detections with text
    for x, y, x2, y2, w, h, conf, cls, age, alpha, color in recent_detections:
        # Only blend the ROI for the bbox
        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            continue
        overlay = roi.copy()
        thickness = 3  # Thicker for recent detections
        cv2.rectangle(overlay, (0, 0), (w, h), color, thickness)
        blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
        frame[y:y2, x:x2] = blended

        # Draw text for recent detections
        cv2.putText(
            frame,
            f'{class_names.get(cls, str(cls))} {conf:.2f}',
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame

if not hasattr(draw_yolo_detections, "bbox_history"):
    draw_yolo_detections.bbox_history = []