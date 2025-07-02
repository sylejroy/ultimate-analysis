from ultralytics import YOLO


model = YOLO("finetune/field_finder_yolo11n-seg/segmentation_finetune/weights/best.pt")

def run_field_segmentation(frame):
    results = model.predict(frame, verbose=False, imgsz=960)
    return results  # Return the results object