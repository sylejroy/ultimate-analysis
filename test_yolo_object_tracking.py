import cv2
import numpy as np
from ultralytics import YOLO
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from random import shuffle

# Helper function to reset tracker
def reset_tracker():
    global tracker
    tracker = DeepSort(max_age=3)
    draw_track_history.track_histories = {}  # Reset track histories

def draw_yolo_detections(frame, detections):
    # You may want to define your class names here or import them if available
    class_names = model.names if hasattr(model, 'names') else {}
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

def draw_track_history(frame, tracks, history_length=300):
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

# Assign a unique color for each track_id
def get_color(idx):
    np.random.seed(idx)
    color = tuple(int(x) for x in np.random.randint(0, 255, 3))

    return color

    # Estimate camera motion using feature matching and homography (perspective)

def draw_motion_vector(frame, H):
    """
    Draws the camera motion vector on the frame using the homography matrix H.
    The vector is drawn from the center of the frame to the transformed center.
    """
    if H is None:
        return
    h, w = frame.shape[:2]
    center = np.array([[w // 2, h // 2]], dtype=np.float32).reshape(-1, 1, 2)
    center_transformed = 2*cv2.perspectiveTransform(center, H)
    pt1 = tuple(center[0, 0].astype(int))
    pt2 = tuple(center_transformed[0, 0].astype(int))
    cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 3, tipLength=0.2)


def estimate_camera_motion(prev_frame, curr_frame):
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None
    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 8:
        return None, None
    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Find homography (perspective transform)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum() if mask is not None else 0
    return H, inliers



# Paths
weights_path = "finetune/object_detection_yolo11l/finetune3/weights/best.pt"
input_dir = "input/dev_data"

# Load YOLO model
model = YOLO(weights_path)

# Initialize DeepSort tracker
reset_tracker()

# Get video files with 'snippet' in the name
video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
               if 'snippet' in f and f.lower().endswith(('.mp4', '.avi', '.mov'))]
shuffle(video_files)  # Shuffle the video files for random processing
idx = 0

while 0 <= idx < len(video_files):
    cap = cv2.VideoCapture(video_files[idx])
    print(f"Processing: {video_files[idx]}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(frame, verbose=False, imgsz=960)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        
        # Call this after YOLO inference and before drawing tracks
        #draw_yolo_detections(frame, detections)

        # DeepSort expects: [ [x, y, w, h], confidence, class ]
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracks with unique colors
        for track in tracks:

            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)


            color = get_color(int(track_id))

            conf = track.det_conf if hasattr(track, 'det_conf') else None
            if conf is not None:
                cv2.putText(
                    frame,
                    f'Conf: {conf:.2f}',
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            class_name = model.names.get(track.det_class, str(track.det_class)) if hasattr(track, 'det_class') else ''
            if class_name:
                cv2.putText(
                    frame,
                    f'Class: {class_name}',
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw track history with unique colors
        draw_track_history(frame, tracks, history_length=300)
        # Estimate camera motion
        if 'prev_frame' in locals():
            H, inliers = estimate_camera_motion(prev_frame, frame)
            draw_motion_vector(frame, H)
        prev_frame = frame.copy()
        cv2.imshow("YOLO + DeepSort Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            idx += 1
            reset_tracker()
            break
        elif key == ord('b'):
            idx = max(0, idx - 1)
            reset_tracker()
            break
        elif key == ord('r'):
            reset_tracker()
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
    cap.release()
cv2.destroyAllWindows()