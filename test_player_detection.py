import cv2
import os
from ultralytics import YOLO
import itertools

# Path to your ultimate-analysis dataset in YOLO format
DATASET_PATH = "training_data/object_detection.v3i.yolov8/data.yaml"
MODEL_STRING = 'yolo11l'  # base model to use & fine-tune
MODEL_PATH = 'object_detection_' + MODEL_STRING + '/finetune3/weights/best.pt'

def train_model():
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO(MODEL_STRING + '.pt')
    # Fine-tune the model on your ultimate-analysis dataset
    model.train(
        data=DATASET_PATH,
        epochs=400,
        imgsz=960,
        batch=0.8,  # Set a reasonable batch size for most GPUs
        patience=200,
        project='object_detection_' + MODEL_STRING,
        name='finetune'
    )
def visualize_results():
    # Load the best model after training
    best_model = YOLO(MODEL_PATH)
    input_dir = "input/dev_data"
    # Gather all video files containing "snippet" in the name
    video_files = [
        f for f in os.listdir(input_dir)
        if "snippet" in f and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    #video_files.sort()  # Optional: sort for consistent order

    if not video_files:
        print("No snippet videos found.")
        return

    idx = 0
    while True:
        filename = video_files[idx]
        video_path = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(video_path)
        print(f"Processing {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the frame
            results = best_model.predict(frame, imgsz=960, conf=0.25)
            best_model.predict()

            # Visualize results on the frame
            annotated_frame = results[0].plot(line_width=1, font_size=0.2)

            # Display the frame
            cv2.imshow(f'Detection Results - {filename}', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):  # 'n' for next video
                break
            elif key == ord('b'):  # 'b' for previous video
                break

        cap.release()
        cv2.destroyAllWindows()

        # Handle navigation with 'n' and 'b'
        if key == ord('n'):
            idx = (idx + 1) % len(video_files)
        elif key == ord('b'):
            idx = (idx - 1) % len(video_files)
        else:
            # If not 'n', 'b', or 'q', stay on current video
            pass


if __name__ == "__main__":
    #train_model()
    visualize_results()