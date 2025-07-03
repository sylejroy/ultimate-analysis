from ultralytics import YOLO
import easyocr
import cv2
import numpy as np

# ---- CONFIGURABLE DEFAULT ----
DEFAULT_PLAYER_ID_METHOD = "easyocr"  # Change to "yolo" to use YOLO by default
# ------------------------------

# Global state
player_id_model = None
player_id_model_path = None
player_id_method = DEFAULT_PLAYER_ID_METHOD
easyocr_reader = None

def set_player_id_method(method):
    global player_id_method
    player_id_method = method.lower()
    print(f"[DEBUG] set_player_id_method: {player_id_method}")

def load_player_id_model(path):
    global player_id_model, player_id_model_path
    print(f"[DEBUG] Loading player ID model from: {path}")
    player_id_model_path = path
    player_id_model = YOLO(player_id_model_path)

def set_player_id_model(path):
    set_player_id_method("yolo")
    load_player_id_model(path)

def set_easyocr():
    global easyocr_reader
    set_player_id_method("easyocr")
    if easyocr_reader is None:
        print("[DEBUG] Initializing EasyOCR reader")
        easyocr_reader = easyocr.Reader(['en'], gpu=True)


def run_player_id(frame):
    global easyocr_reader
    if player_id_method == "yolo":
        if player_id_model is None:
            raise RuntimeError("Player ID YOLO model not loaded.")
        results = player_id_model(frame, verbose=False, imgsz=640, conf=0.5)
        digits = []
        for result in results:
            d_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            d_classes = result.boxes.cls.cpu().numpy().astype(int)
            # Sort digits left-to-right
            digits = sorted(zip(d_boxes, d_classes), key=lambda x: x[0][0])
        digit_str = ''.join(str(d[1]) for d in digits)
        return digit_str, digits
    elif player_id_method == "easyocr":
        if easyocr_reader is None:
            set_easyocr()
        # Only blur before OCR, no other preprocessing
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        ocr_results = easyocr_reader.readtext(blurred, detail=1, allowlist='0123456789', decoder='greedy')
        ocr_boxes = [res[0] for res in ocr_results]
        digit_str = ''.join([res[1] for res in ocr_results])
        return digit_str, ocr_boxes
    else:
        raise RuntimeError("Unknown player ID method.")

# Initialize the default method at startup
if DEFAULT_PLAYER_ID_METHOD == "easyocr":
    try:
        set_easyocr()
    except Exception as e:
        print(f"[DEBUG] Could not initialize EasyOCR at startup: {e}")