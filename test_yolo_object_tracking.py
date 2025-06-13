import cv2
import numpy as np
from ultralytics import YOLO
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths
weights_path = "finetune/object_detection_yolo11l/finetune3/weights/best.pt"
input_dir = "input/dev_data"

# Load YOLO model
model = YOLO(weights_path)

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# Get video files with 'snippet' in the name
video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
               if 'snippet' in f and f.lower().endswith(('.mp4', '.avi', '.mov'))]
video_files.sort()
idx = 0

while 0 <= idx < len(video_files):
    cap = cv2.VideoCapture(video_files[idx])
    print(f"Processing: {video_files[idx]}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # DeepSort expects: [ [x, y, w, h], confidence, class ]
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.putText(frame, f'ID {track_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)

        cv2.imshow("YOLO + DeepSort Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            idx += 1
            break
        elif key == ord('b'):
            idx = max(0, idx - 1)
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
    cap.release()
cv2.destroyAllWindows()