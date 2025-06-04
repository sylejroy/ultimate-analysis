from ultralytics import YOLO
import cv2
import os

# Path to your dataset in YOLO format (should contain images, labels, and data.yaml)
DATASET_PATH = "training_data/field finder.v2i.yolov8/data.yaml"
VIDEO_PATH = "input/portland_vs_san_francisco_snippet_43857.mp4"
MODEL_STRING = 'yolo11n-seg' # base model to use & fine-tune
MODEL_PATH = 'field_finder_' + MODEL_STRING + '/segmentation_finetune/weights/best.pt'

def train_model():
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO(MODEL_STRING + '.pt')
    # Fine-tune the model on your dataset
    model.train(
        data=DATASET_PATH,
        epochs=50,            # Adjust epochs as needed
        imgsz=640,            # Image size
        batch=0.90,           # Adjust batch size based on your GPU
        patience=10,          # Early stopping
        project='field_finder_'+ MODEL_STRING,
        name='segmentation_finetune'
    )

def visualize_results():
    # Load the best model after training
    best_model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = best_model.predict(frame, imgsz=640, conf=0.55)

        # Visualize results on the frame
        annotated_frame = results[0].plot()
        print(results[0])

        # Display the frame
        cv2.imshow('YOLOv8 Segmentation Results', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# annotate training data
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
    #train_model()
    visualize_results()
    #annotate_training_data()
    #auto_annotate_training_data()