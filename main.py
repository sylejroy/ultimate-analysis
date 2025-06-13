import cv2
import random
from ultralytics import YOLO
import os
import numpy as np

INPUT_FOLDER = "input/dev_data"  # Change as needed
YOLO_PLAYERS_MODEL = "yolo11l"  # Path to fine-tuned player/disc model
YOLO_FIELD_MODEL = "yolo11m-seg"  # Path to fine-tuned field segmentation model
YOLO_PLAYERS_MODEL_PATH = "finetune/object_detection_" + YOLO_PLAYERS_MODEL + "/finetune3/weights/best.pt"
YOLO_FIELD_MODEL_PATH = "finetune/field_finder_" + YOLO_FIELD_MODEL + "/segmentation_finetune/weights/best.pt"

FIELD_CONF = 0.8
PLAYER_CONF = 0.3

def visualize_inference():
    snippet_files = sorted([f for f in os.listdir(INPUT_FOLDER) if "snippet" in f])
    if not snippet_files:
        print("No snippet files found.")
        return

    yolo_players = YOLO(YOLO_PLAYERS_MODEL_PATH)  # Path to fine-tuned player/disc model
    yolo_field = YOLO(YOLO_FIELD_MODEL_PATH)  # Path to fine-tuned field segmentation model

    idx = 0
    while 0 <= idx < len(snippet_files):
        snippet_path = os.path.join(INPUT_FOLDER, snippet_files[idx])
        cap = cv2.VideoCapture(snippet_path)
        print(f"Processing: {snippet_files[idx]}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect players and discs
            results_players = yolo_players(frame, imgsz=960, conf=PLAYER_CONF)
            for box in results_players[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # Get class_id and label before using them
                if hasattr(results_players[0], "boxes") and hasattr(results_players[0].boxes, "cls"):
                    class_id = int(results_players[0].boxes.cls[results_players[0].boxes.xyxy.tolist().index(box.tolist())])
                    label = results_players[0].names[class_id] if hasattr(results_players[0], "names") else str(class_id)
                else:
                    class_id = 0
                    label = "unknown"
                # Add confidence if available
                if hasattr(results_players[0].boxes, "conf"):
                    conf = float(results_players[0].boxes.conf[results_players[0].boxes.xyxy.tolist().index(box.tolist())])
                    label += f" {conf:.2f}"
                
                # Assign a unique color for each class
                # Use HSV color space to generate visually distinct colors
                num_classes = len(results_players[0].names) if hasattr(results_players[0], "names") else 10
                hue = int(180 * class_id / max(num_classes, 1))
                color_hsv = cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                )[0][0]
                color = tuple(int(c) for c in color_hsv)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            
            # Semantic segmentation for field
            
            results_field = yolo_field(frame, imgsz=640, conf=FIELD_CONF)
            if hasattr(results_field[0], "masks") and results_field[0].masks is not None:
                masks = results_field[0].masks.data.cpu().numpy()
                if hasattr(results_field[0], "boxes") and hasattr(results_field[0].boxes, "cls"):
                    class_ids = results_field[0].boxes.cls.cpu().numpy().astype(int)
                else:
                    class_ids = [0] * len(masks)
                if hasattr(results_field[0], "names"):
                    class_names = results_field[0].names
                else:
                    class_names = {i: str(i) for i in range(10)}
                for i, mask in enumerate(masks):
                    class_id = class_ids[i] if i < len(class_ids) else 0
                    hue = int(180 * class_id / max(len(class_names), 1))
                    color_hsv = cv2.cvtColor(
                        np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                    )[0][0]
                    color = tuple(int(c) for c in color_hsv)
                    overlay = frame.copy()
                    # Resize mask to match frame size
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    kernel = np.ones((10, 10), np.uint8)
                    mask_resized = cv2.erode(mask_resized, kernel, iterations=2)
                    mask_resized = cv2.dilate(mask_resized, kernel, iterations=2)
                    
                    mask_bool = mask_resized > 0.1

                    contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    

                    for contour in contours:
                        # Simplify contours using approxPolyDP
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                        cv2.drawContours(frame, [simplified_contour], -1, color, 2)
                    
                    
                    overlay[mask_bool] = (
                        0.5 * overlay[mask_bool] + 0.5 * np.array(color)
                    ).astype(np.uint8)

                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                   

                    # Draw bounding box and label
                    
                    if hasattr(results_field[0], "boxes") and i < len(results_field[0].boxes.xyxy):
                        x1, y1, x2, y2 = map(int, results_field[0].boxes.xyxy[i])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        label = class_names[class_id] if class_id in class_names else str(class_id)
                        conf = float(results_field[0].boxes.conf[i])
                        label += f" {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            
            cv2.imshow("Inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
            elif key == ord('b'):
                idx = max(idx - 2, -1)  # -1 because idx will be incremented below
                break

        cap.release()
        idx += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_inference()


