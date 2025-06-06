from ultralytics import YOLO
import cv2
import os

# Path to your dataset in YOLO format (should contain images, labels, and data.yaml)
DATASET_PATH = "training_data/field finder.v2i.yolov8/data.yaml"
MODEL_STRING = 'yolo11x-seg' # base model to use & fine-tune
MODEL_PATH = 'field_finder_' + MODEL_STRING + '/segmentation_finetune/weights/best.pt'

def train_model():
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO(MODEL_STRING + '.pt')
    # Fine-tune the model on your dataset
    model.train(
        data=DATASET_PATH,
        epochs=70,            # Adjust epochs as needed
        imgsz=640,            # Image size
        batch=0.90,           # Adjust batch size based on your GPU
        patience=20,          # Early stopping
        project='field_finder_'+ MODEL_STRING,
        name='segmentation_finetune'
    )

def visualize_results():
    # Find all video files containing "snippet" in the name
    input_dir = "input"
    video_files = [f for f in os.listdir(input_dir) if "snippet" in f and f.endswith('.mp4')]
    video_files.sort()
    if not video_files:
        print("No snippet videos found.")
        return

    best_model = YOLO(MODEL_PATH)
    idx = 0

    while 0 <= idx < len(video_files):
        video_path = os.path.join(input_dir, video_files[idx])
        cap = cv2.VideoCapture(video_path)
        print(f"Playing: {video_files[idx]} ({idx+1}/{len(video_files)})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Run inference on the frame
            results = best_model.predict(frame, imgsz=640, conf=0.65)
            annotated_frame = results[0].plot(line_width=1, font_size=0.2)
            cv2.imshow('Segmentation Results', annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                idx += 1
                break
            elif key == ord('b'):
                idx -= 1
                break

        cap.release()
        cv2.destroyAllWindows()
        # Clamp idx to valid range
        if idx < 0:
            idx = 0
        if idx >= len(video_files):
            break

def annotate_training_data():
    # Load the best model after training
    best_model = YOLO(MODEL_PATH)
    # Path to training data frames in training_data folder
    training_data_path = "training_data/sampled_frames"
    frame_files = [f for f in os.listdir(training_data_path) if f.endswith('.jpg')]

    # Annotate each frame in yolov8 format including TXT annotations and YAML config
    for frame_file in frame_files:
        frame_path = os.path.join(training_data_path, frame_file)
        frame = cv2.imread(frame_path)
        results = best_model.predict(frame, imgsz=640, conf=0.75)
        results[0].show()



if __name__ == "__main__":
    # Uncomment as needed
    #bqtrain_model()
    visualize_results()