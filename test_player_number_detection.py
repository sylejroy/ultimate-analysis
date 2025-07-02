# finetune_yolo11l_digits.py

from ultralytics import YOLO
import cv2
import glob
import os
import numpy as np
import easyocr

# Global parameters for paths
DATA_YAML_PATH = r'training_data\digits.v1i.yolov8\data.yaml'
BASE_MODEL_PATH = 'yolo11l'
OBJECT_MODEL_PATH = r'finetune\object_detection_yolo11l\finetune3\weights\best.pt'
DIGIT_MODEL_PATH = 'finetune/digit_detector_' + BASE_MODEL_PATH + '/finetune/weights/best.pt'
VIDEO_DIR = r'input\dev_data'

def train_yolo_digits():
    """
    Fine-tune YOLO large model on the digits dataset.
    """
    # Use global parameters
    data_yaml = DATA_YAML_PATH
    model = YOLO(BASE_MODEL_PATH + '.pt')  # Load the base model

    # Train the model
    model.train(
        data=data_yaml,
        epochs=100,          # Adjust epochs as needed
        imgsz=640,           # Image size
        batch=0.8,           # Batch size
        patience=40,
        project='finetune/digit_detector_' + BASE_MODEL_PATH,
        name='finetune'
    )

def run_object_and_digit_detection():
    # Load models
    object_model = YOLO(OBJECT_MODEL_PATH)
    digit_model = YOLO(DIGIT_MODEL_PATH)

    # Find video files with "snippet" in the name
    video_files = glob.glob(os.path.join(VIDEO_DIR, '*snippet*.mp4'))
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frame = frame.copy()  # Keep the original for detection
            player_class_id = 1  # Assuming player class ID is 1, adjust if necessary

            # Run detection on raw_frame, not frame
            results = object_model(raw_frame, verbose=False, imgsz=960)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for box, cls in zip(boxes, classes):
                    if cls != player_class_id:
                        continue
                    x1, y1, x2, y2 = box
                    # Use only the top half of the detected bounding box
                    mid_y = y1 + (y2 - y1) // 2
                    obj_crop = raw_frame[y1:mid_y, x1:x2]
                    if obj_crop.size == 0:
                        continue
            

                    # Display every 10th cropped object in a separate frame, stretched by a factor of 10
                    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % 10) == 0:
                        h, w = obj_crop.shape[:2]
                        stretched_crop = cv2.resize(obj_crop, (w * 10, h * 10), interpolation=cv2.INTER_NEAREST)
                        # Prepare the size text
                        size_text = f"{x2 - x1}x{mid_y - y1}"
                        # Put the size text in the bottom right corner
                        text_size, _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_x = stretched_crop.shape[1] - text_size[0] - 10
                        text_y = stretched_crop.shape[0] - 10
                        cv2.putText(
                            stretched_crop,
                            size_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow('Cropped Object', stretched_crop)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Run digit detection on the cropped object
                    digit_results = digit_model(obj_crop, verbose=False, imgsz=640, conf=0.5)
                    digit_str = ""
                    for d_result in digit_results:
                        d_boxes = d_result.boxes.xyxy.cpu().numpy().astype(int)
                        d_classes = d_result.boxes.cls.cpu().numpy().astype(int)

                        # Sort digits left-to-right
                        digits = sorted(zip(d_boxes, d_classes), key=lambda x: x[0][0])
                        digit_str = ''.join(str(d[1]) for d in digits)

                        # Draw digit boxes (optional)
                        for db, dc in digits:
                            dx1, dy1, dx2, dy2 = db
                            cv2.rectangle(obj_crop, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1)

                    # Draw object bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Calculate and draw average confidence of detected digits
                    if digit_results and len(digits) > 0:
                        confidences = d_result.boxes.conf.cpu().numpy()
                        avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
                        conf_text = f"{avg_conf:.2f}"
                        cv2.putText(
                            frame,
                            conf_text,
                            (x2 + 10, y1 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 200, 0),
                            2
                        )
                    # Put digits to the right of the bounding box
                    if digit_str:
                        cv2.putText(frame, digit_str, (x2 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(f'Detection: {os.path.basename(video_path)}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def run_object_and_digit_detection_easyocr():
    # Load object detection model
    object_model = YOLO(OBJECT_MODEL_PATH)
    # Initialize EasyOCR reader (English, GPU if available)
    reader = easyocr.Reader(['en'], gpu=True)

    # Find video files with "snippet" in the name
    video_files = glob.glob(os.path.join(VIDEO_DIR, '*snippet*.mp4'))
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            player_class_id = 1
            results = object_model(frame, verbose=False, imgsz=960)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for box, cls in zip(boxes, classes):
                    if cls != player_class_id:
                        continue
                    x1, y1, x2, y2 = box
                    mid_y = y1 + (y2 - y1) // 2
                    obj_crop = frame[y1:mid_y, x1:x2]
                    if obj_crop.size == 0:
                        continue

                    # Display every 10th cropped object in a separate frame, stretched by a factor of 10
                    # if (cap.get(cv2.CAP_PROP_POS_FRAMES) % 10) == 0:
                    #     h, w = obj_crop.shape[:2]
                    #     stretched_crop = cv2.resize(obj_crop, (w * 10, h * 10), interpolation=cv2.INTER_NEAREST)
                    #     size_text = f"{x2 - x1}x{mid_y - y1}"
                    #     text_size, _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    #     text_x = stretched_crop.shape[1] - text_size[0] - 10
                    #     text_y = stretched_crop.shape[0] - 10
                    #     cv2.putText(
                    #         stretched_crop,
                    #         size_text,
                    #         (text_x, text_y),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         1,
                    #         (0, 255, 0),
                    #         2
                    #     )
                    #     cv2.imshow('Cropped Object', stretched_crop)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break

                    # Use EasyOCR to detect digits in the cropped image
                    digit_str = ""
                    if obj_crop.size != 0:
                        # Convert to RGB for EasyOCR
                        crop_rgb = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB)
                        ocr_results = reader.readtext(crop_rgb, detail=0, allowlist='0123456789', min_size=5, rotation_info=[0, 90, 180 ,270], text_threshold=0.3)
                        # Join all detected numbers
                        digit_str = ''.join(ocr_results)
                    # Draw object bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Put digits to the right of the bounding box
                    if digit_str:
                        cv2.putText(frame, digit_str, (x2 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Detection (EasyOCR)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    #train_yolo_digits()
    run_object_and_digit_detection()
    #run_object_and_digit_detection_easyocr()