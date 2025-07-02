from ultralytics import YOLO

# Load YOLO model once (global)
weights_path = "finetune/object_detection_yolo11l/finetune3/weights/best.pt"
model = YOLO(weights_path)

def run_inference(frame):
    """
    Runs YOLO inference on the given frame and returns detections in the format:
    [ ([x, y, w, h], conf, cls), ... ]
    """
    results = model.predict(frame, verbose=False, imgsz=960)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
    return detections

def get_class_names():
    return model.names if hasattr(model, 'names') else {}