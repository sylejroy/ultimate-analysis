from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from collections import deque
import random

# Tracker selection
tracker_type =  "deepsort" #"histogram" # or "deepsort"
tracker = None

# --- Histogram-based tracker implementation ---

MAX_TRACK_LOST = 10
MAX_HISTORY = 300

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def hist_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def get_histogram(img, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((180,))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

class Track:
    def __init__(self, track_id, bbox, hist, frame_id, det_class=None):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.hist = hist
        self.last_frame = frame_id
        self.history = deque([bbox], maxlen=MAX_HISTORY)
        self.lost = 0
        self.confirmed = True
        self.det_class = det_class  # <-- Store class

    def update(self, bbox, hist, frame_id):
        self.bbox = bbox
        self.hist = hist
        self.last_frame = frame_id
        self.history.append(bbox)
        self.lost = 0

    def is_confirmed(self):
        return True

    def to_ltrb(self):
        return self.bbox

class HistogramTracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 0
        self.frame_id = 0

    def reset(self):
        self.tracks = []
        self.next_track_id = random.randint(0, 1000000)
        self.frame_id = 0

    def update(self, frame, detections):
        # detections: [([x, y, w, h], conf, cls), ...]
        det_bboxes = []
        det_hists = []
        det_classes = []
        for det in detections:
            (x, y, w, h), conf, cls = det
            x1, y1, x2, y2 = x, y, x + w, y + h
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            hist = get_histogram(frame, bbox)
            det_bboxes.append(bbox)
            det_hists.append(hist)
            det_classes.append(cls)

        assigned = set()
        for track in self.tracks:
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
                track.update(det_bboxes[min_idx], det_hists[min_idx], self.frame_id)
                assigned.add(min_idx)
            else:
                track.lost += 1

        # Create new tracks for unassigned detections
        for i, (bbox, hist) in enumerate(zip(det_bboxes, det_hists)):
            if i not in assigned:
                self.tracks.append(Track(self.next_track_id, bbox, hist, self.frame_id, det_class=det_classes[i]))
                self.next_track_id += 1

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.lost <= MAX_TRACK_LOST]

        self.frame_id += 1
        return self.tracks

# --- Tracker selection logic ---

def set_tracker_type(t_type):
    global tracker_type, tracker
    tracker_type = t_type
    print(f"[TRACKING DEBUG] Setting tracker type to: {tracker_type}")
    if tracker_type == "deepsort":
        tracker = DeepSort(max_age=10, embedder="mobilenet", n_init=5)
        print("[TRACKING DEBUG] DeepSort tracker initialized.")
    elif tracker_type == "histogram":
        tracker = HistogramTracker()
        print("[TRACKING DEBUG] Histogram tracker initialized.")

set_tracker_type(tracker_type)

def run_tracking(frame, detections):
    global tracker
    if tracker_type == "deepsort":
        return tracker.update_tracks(detections, frame=frame)
    elif tracker_type == "histogram":
        return tracker.update(frame, detections)
    return []

def reset_tracker():
    global tracker
    set_tracker_type(tracker_type)  # This re-initializes the tracker
    # Randomize next_track_id if using HistogramTracker
    if tracker_type == "histogram" and hasattr(tracker, 'next_track_id'):
        tracker.next_track_id = random.randint(0, 1000000)