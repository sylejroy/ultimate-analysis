import cv2
import os
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolo11n.pt') # n, s, m, l, x in order of size

# Get a snippet video file from the input folder
input_folder = 'input'
video_files = [f for f in os.listdir(input_folder) 
               if f.lower().endswith(('.mp4', '.avi', '.mov')) 
               and 'snippet' in f.lower()]

if not video_files:
    raise FileNotFoundError("No video snippet files found in the input folder. "
                          "Please ensure you have short video files with 'snippet' in their name.")

# Print available snippets and select the first one
print("Available video snippets:")
for i, file in enumerate(video_files):
    print(f"{i+1}. {file}")
video_path = os.path.join(input_folder, video_files[0])
print(f"\nUsing: {video_files[0]}")

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video file")
    exit()

while True:
    # Read a frame from the video
    frame_read_success, frame = video_capture.read()
    
    # If frame is read correctly, frame_read_success is True
    if not frame_read_success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    # Run object detection on the frame
    results = model(frame)
    
    # Visualize results on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            
            # Get class details
            class_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = model.names[class_id]
            
            # Calculate box dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare text with details
            label = f'{class_name} {conf:.2f}'
            dimensions = f'{width}x{height}'
            
            # Add background for text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1-label_height-5), (x1+label_width, y1), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, dimensions, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow('Video Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
video_capture.release()
cv2.destroyAllWindows()