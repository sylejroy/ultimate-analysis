# sample the video input\portland_vs_san_francisco_2024.mp4 once every 5 seconds and output to a folder called "training data"
import cv2
import os

def sample_video_frames(video_path, output_folder, interval=5):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Save the frame every `frame_interval` frames
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)

        # Print progress based on total frames
        print(f"\rProgress: {frame_count/total_frames*100:.2f}%", end="")

        frame_count += 1

    cap.release()

# Usage
video_path = 'input/portland_vs_san_francisco_2024.mp4'
output_folder = 'training_data'
sample_video_frames(video_path, output_folder)

