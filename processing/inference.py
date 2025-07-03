from ultralytics import YOLO

# Global model and path
model = None
weights_path = None

def load_model(path):
    global model, weights_path
    print(f"[DEBUG] Loading detection model from: {path}")
    weights_path = path
    model = YOLO(weights_path)

# Load default model at startup
load_model("finetune/object_detection_yolo11l/finetune3/weights/best.pt")

def set_detection_model(path):
    """Set and reload the detection model at runtime."""
    print(f"[DEBUG] set_detection_model called with: {path}")
    load_model(path)

def run_inference(frame):
    """
    Runs YOLO inference on the given frame and returns detections in the format:
    [ ([x, y, w, h], conf, cls), ... ]
    """
    if model is None:
        raise RuntimeError("YOLO model is not loaded.")
    results = model.predict(frame, verbose=False, imgsz=960)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
    return detections

def get_class_names():
    return model.names if model is not None and hasattr(model, 'names') else {}