import cv2
from processing.inference import get_class_names

def draw_yolo_detections(frame, detections):
    # Optionally avoid modifying the original frame
    # frame = frame.copy()
    class_names = get_class_names()
    for det in detections:
        (x, y, w, h), conf, cls = det
        x2, y2 = x + w, y + h
        class_name = class_names.get(cls, str(cls))
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            frame,
            f'{class_name} {conf:.2f}',
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )
    return frame