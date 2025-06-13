import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# Load YOLO11 model
model_path = Path("finetune/object_detection_yolo11l/finetune3/weights/best.pt")
model = YOLO(model_path)

# Video input
video_path = "input/dev_data/san_francisco_vs_colorado_2024_snippet_1_67648.mp4"
cap = cv2.VideoCapture(video_path)

# Tracker parameters
MAX_TRACK_LOST = 10
MAX_HISTORY = 300

class Track:
    def __init__(self, track_id, bbox, hist, frame_id):
        self.track_id = track_id
        self.bbox = bbox
        self.hist = hist
        self.last_frame = frame_id
        self.history = deque([bbox], maxlen=MAX_HISTORY)
        self.lost = 0

    def update(self, bbox, hist, frame_id):
        self.bbox = bbox
        self.hist = hist
        self.last_frame = frame_id
        self.history.append(bbox)
        self.lost = 0

def get_histogram(img, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((180,))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def hist_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

tracks = []
next_track_id = 0
frame_id = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    detections = results[0].boxes.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    det_bboxes = []
    det_hists = []
    for det in detections:
        xyxy = det.xyxy[0]  # [x1, y1, x2, y2]
        conf = det.conf[0]  # Confidence score
        cls = det.cls[0]  # Class ID (not used here)
        x1, y1, x2, y2 = xyxy
        if conf < 0.3:
            continue
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        hist = get_histogram(frame, bbox)
        det_bboxes.append(bbox)
        det_hists.append(hist)

    # Association
    assigned = set()
    for track in tracks:
        min_dist = float('inf')
        min_idx = -1
        for i, (bbox, hist) in enumerate(zip(det_bboxes, det_hists)):
            if i in assigned:
                continue
            if iou(track.bbox, bbox) < 0.1:
                continue
            dist = hist_distance(track.hist, hist)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx != -1 and min_dist < 0.5:
            track.update(det_bboxes[min_idx], det_hists[min_idx], frame_id)
            assigned.add(min_idx)
        else:
            track.lost += 1

    # Create new tracks for unassigned detections
    for i, (bbox, hist) in enumerate(zip(det_bboxes, det_hists)):
        if i not in assigned:
            tracks.append(Track(next_track_id, bbox, hist, frame_id))
            next_track_id += 1

    # Remove lost tracks
    tracks = [t for t in tracks if t.lost <= MAX_TRACK_LOST]

    # Visualization
    for track in tracks:
        color = (int(track.track_id * 37) % 256, int(track.track_id * 17) % 256, int(track.track_id * 97) % 256)
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID:{track.track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Draw track history
        for i in range(1, len(track.history)):
            pt1 = ((track.history[i-1][0]+track.history[i-1][2])//2, (track.history[i-1][1]+track.history[i-1][3])//2)
            pt2 = ((track.history[i][0]+track.history[i][2])//2, (track.history[i][1]+track.history[i][3])//2)
            cv2.line(frame, pt1, pt2, color, 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()