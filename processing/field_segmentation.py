from ultralytics import YOLO

# Global model and path
field_model = None
field_model_path = None

def load_field_model(path):
    global field_model, field_model_path
    print(f"[DEBUG] Loading field segmentation model from: {path}")
    field_model_path = path
    field_model = YOLO(field_model_path)

# Load a default field segmentation model at startup
load_field_model("finetune/field_finder_yolo11x-seg/segmentation_finetune4/weights/best.pt")

def set_field_model(path):
    """Set and reload the field segmentation model at runtime."""
    print(f"[DEBUG] set_field_model called with: {path}")
    load_field_model(path)

def run_field_segmentation(frame):
    """
    Runs YOLO segmentation on the given frame and returns the results.
    """
    if field_model is None:
        raise RuntimeError("Field segmentation model is not loaded.")
    results = field_model.predict(frame, verbose=False, imgsz=960)
    return results  # Return the results object