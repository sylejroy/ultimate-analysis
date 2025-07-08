import cv2
import numpy as np
from ultralytics import YOLO
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from random import shuffle
from ultralytics import YOLO as DigitYOLO

# Helper function to reset tracker
def reset():
    global tracker
    tracker = DeepSort(max_age=10, embedder="mobilenet", n_init=5)
    draw_track_history.track_histories = {}  # Reset track histories
    global frame_list
    frame_list = []  # Reset frame list

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

def estimate_camera_motion(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features in both frames
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)
    

    # Calculate homography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M

# display the camera motion on the frame as an arrow
def draw_camera_motion(frame, motion_matrix):
    if motion_matrix is None:
        return frame

    # Define the center of the frame
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    # Define a point to represent the motion direction
    motion_point = np.array([[center[0] + 50, center[1]]], dtype=np.float32).reshape(-1, 1, 2)

    # Apply the motion matrix to the point
    transformed_point = cv2.perspectiveTransform(motion_point, motion_matrix)

    # Draw an arrow from the center to the transformed point
    cv2.arrowedLine(frame, center, (int(transformed_point[0][0][0]), int(transformed_point[0][0][1])), (255, 0, 0), 2, tipLength=0.1)

def detect_and_draw_digits(frame, yolo_detections, digit_model_path="finetune/digit_detector/finetune/weights/best.pt"):
    # Load digit detector model only once
    if not hasattr(detect_and_draw_digits, "digit_model"):
        detect_and_draw_digits.digit_model = DigitYOLO(digit_model_path)
    digit_model = detect_and_draw_digits.digit_model
    for det in yolo_detections:
        (x, y, w, h), conf, cls = det
        # Focus on the top half of the bounding box
        top_half_y2 = y + h // 2
        top_half = frame[y:top_half_y2, x:x + w]
        if top_half.size == 0:
            continue
        # Run digit detector on the top half
        digit_results = digit_model.predict(top_half, verbose=False, imgsz=256)[0]
        digit_boxes = []
        for box in digit_results.boxes:
            dx1, dy1, dx2, dy2 = map(int, box.xyxy[0])
            dconf = float(box.conf[0])
            dcls = int(box.cls[0])
            digit_boxes.append((dx1, dy1, dx2, dy2, dconf, dcls))
        # Sort digits left to right
        digit_boxes.sort(key=lambda b: b[0])
        digits = [str(b[5]) for b in digit_boxes]
        # Draw the top half bounding box
        cv2.rectangle(frame, (x, y), (x + w, top_half_y2), (0, 255, 255), 2)
        # Draw detected digits on the right side of the YOLO box
        if digits:
            text = "".join(digits)
            text_x = x + w + 10
            text_y = y + 25
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


# Paths
weights_path = "finetune/object_detection_yolo11l/finetune3/weights/best.pt"
input_dir = "input/dev_data"

# Load YOLO model
model = YOLO(weights_path)

# Initialize DeepSort tracker
reset()

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

        frame_list.append(frame)  # Store the frame for later use
        
        # Draw legend with keybinds in the top left corner
        legend = [
            "Keybinds:",
            "n - Next video",
            "b - Previous video",
            "r - Reset tracker",
            "q - Quit",
            "Space - Pause/Unpause",
        ]
        # Draw opaque background for the legend
        legend_width = 260
        legend_height = 22 * len(legend) + 10
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (5, 5),
            (5 + legend_width, 5 + legend_height),
            (40, 40, 40),
            thickness=-1
        )
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        x, y0 = 10, 25
        for i, line in enumerate(legend):
            y = y0 + i * 22
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 if i == 0 else 0.6,
                (255, 255, 255) if i == 0 else (200, 200, 200),
                2 if i == 0 else 1,
                cv2.LINE_AA
            )
        
        
        
        cv2.imshow("YOLO + DeepSort Tracking", frame)



        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar to pause
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(' '):  # Unpause on spacebar
                    break
                elif key2 == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
        if key == ord('n'):
            idx += 1
            reset()
            break
        elif key == ord('b'):
            idx = max(0, idx - 1)
            reset()
            break
        elif key == ord('r'):
            reset()
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
    cap.release()
cv2.destroyAllWindows()